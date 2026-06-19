"""Sandboxed subprocess runner for code_gen output.

Per KICKOFF.md §10:
  * CWD is restricted to ``projects/<project_id>/sandbox/``.
  * ``resource`` module limits: CPU time, address space, file descriptors.
  * 30-minute hard timeout per call, killing the entire process group.
  * stdout/stderr captured to ``stdout.log``/``stderr.log`` in a per-call
    log directory; exit code logged.

Network isolation is *not* enforced at the v1 layer. The threat model is
accidental damage to the operator's filesystem (handled by CWD restriction +
WSL containment), not malicious code from an attacker. Network-isolated
execution (network namespaces, iptables) is queued for post-v1 review.

Resource limits use the POSIX ``resource`` module — Linux/WSL/macOS only.
On Windows-native Python the limits silently degrade to "no limits"; do not
run untrusted code outside WSL.
"""

from __future__ import annotations

import contextlib
import os
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import structlog

log = structlog.get_logger("autoscientist.tools.execute")

DEFAULT_TIMEOUT_SECONDS = 30 * 60
DEFAULT_CPU_SECONDS = 30 * 60
DEFAULT_MEMORY_BYTES = 16 * 1024 * 1024 * 1024  # 16 GiB
DEFAULT_NOFILE: int | None = None  # inherit parent unless caller overrides

# Sandbox policy (2026-05-31 audit, item 4). Generated code is supposed to be
# "python train.py"-shaped; these are the only argv[0] basenames the executor
# will launch. Everything else — bash/sh (arbitrary shell), curl/wget/nc
# (exfiltration), make, etc. — is refused so a generated `["bash","-c",...]`
# can't smuggle in a shell or a network fetch. Override per call if a project
# legitimately needs another interpreter.
DEFAULT_ALLOWED_EXECUTABLES: tuple[str, ...] = ("python", "python3", "pytest")

# Wrapper that drops the child into an unprivileged user+network namespace with
# no interfaces, so outbound network is impossible. Works without root (verified
# in WSL). `--` terminates unshare's own option parsing.
_NETNS_WRAPPER: tuple[str, ...] = ("unshare", "--map-root-user", "--net", "--")

# Secret-bearing env vars are scrubbed from the inherited environment before it
# reaches generated, untrusted code (2026-06-18 audit). Otherwise a `python
# train.py` could read ANTHROPIC_API_KEY / GITHUB_PERSONAL_ACCESS_TOKEN /
# BIMCV_TOKEN straight out of os.environ and write them to the sandbox or logs.
# We filter by substring (catches *_API_KEY/*_TOKEN/… without enumerating every
# provider) plus an explicit name backstop; benign vars (PATH, HOME, CUDA_*,
# VIRTUAL_ENV, …) are preserved so normal training code still runs.
_SECRET_ENV_SUBSTRINGS: tuple[str, ...] = (
    "API_KEY", "TOKEN", "SECRET", "PASSWORD", "PASSWD", "CREDENTIAL",
    "ACCESS_KEY", "PRIVATE_KEY", "SESSION_KEY",
)
_SECRET_ENV_NAMES: frozenset[str] = frozenset({
    "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GITHUB_PERSONAL_ACCESS_TOKEN",
    "GH_TOKEN", "BIMCV_TOKEN", "KAGGLE_KEY", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN",
})


def _scrub_secret_env(env: dict[str, str]) -> dict[str, str]:
    """Drop secret-bearing variables from a child environment. Case-insensitive."""
    scrubbed: dict[str, str] = {}
    for key, value in env.items():
        upper = key.upper()
        if key in _SECRET_ENV_NAMES or any(tok in upper for tok in _SECRET_ENV_SUBSTRINGS):
            continue
        scrubbed[key] = value
    return scrubbed


@dataclass
class ExecutionResult:
    exit_code: int
    stdout: str
    stderr: str
    duration_ms: int
    timed_out: bool
    sandbox_dir: str
    log_dir: str
    cmd: list[str] = field(default_factory=list)
    network_isolated: bool = False  # True when the run was wrapped in a netns

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SandboxPolicyError(ValueError):
    """A command was refused by the sandbox policy (shell / disallowed exe)."""


class NetworkIsolationUnavailable(RuntimeError):
    """Network isolation was required but no mechanism (unshare) is available."""


def _ensure_sandbox(project_id: str, projects_root: Path) -> Path:
    sandbox = projects_root / project_id / "sandbox"
    sandbox.mkdir(parents=True, exist_ok=True)
    return sandbox


def _ensure_log_dir(project_id: str, projects_root: Path) -> Path:
    runs = projects_root / project_id / "exec_logs"
    runs.mkdir(parents=True, exist_ok=True)
    nonce = f"{int(time.time() * 1000):x}"
    log_dir = runs / nonce
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


def _clamp_to_hard(resource_module: Any, which: int, requested: int) -> tuple[int, int]:
    """Pick a soft/hard pair we are allowed to set without privileges."""
    _soft, hard = resource_module.getrlimit(which)
    # Unprivileged processes can lower the hard cap but not raise it.
    new_hard = hard if hard > 0 else requested
    if new_hard > 0:
        new_hard = min(new_hard, requested) if requested > 0 else new_hard
    new_soft = min(requested if requested > 0 else new_hard, new_hard) if new_hard > 0 else requested
    return new_soft, new_hard


def _set_resource_limits(
    cpu_seconds: int | None, memory_bytes: int | None, nofile: int | None
) -> None:
    """Called as ``preexec_fn`` in the child process pre-execve.

    The child's session/process group is created by ``start_new_session=True``
    on the Popen call (so a single ``killpg`` reaches the whole tree on
    timeout). This function only sets resource limits.

    Limits are clamped to the parent's hard cap before being applied — an
    unprivileged process cannot raise its hard cap, so a naive ``setrlimit``
    of "16 GiB" can fail on systems whose hard cap is lower than that.
    """
    try:
        import resource
    except ImportError:
        # Windows-native: no resource module. Caller was warned in module docs.
        return

    if cpu_seconds is not None:
        with contextlib.suppress(ValueError, OSError):
            soft, hard = _clamp_to_hard(resource, resource.RLIMIT_CPU, cpu_seconds)
            resource.setrlimit(resource.RLIMIT_CPU, (soft, hard))
    if memory_bytes is not None:
        # macOS / some kernels reject RLIMIT_AS; ignore.
        with contextlib.suppress(ValueError, OSError):
            soft, hard = _clamp_to_hard(resource, resource.RLIMIT_AS, memory_bytes)
            resource.setrlimit(resource.RLIMIT_AS, (soft, hard))
    if nofile is not None:
        with contextlib.suppress(ValueError, OSError):
            soft, hard = _clamp_to_hard(resource, resource.RLIMIT_NOFILE, nofile)
            resource.setrlimit(resource.RLIMIT_NOFILE, (soft, hard))


def execute(
    cmd: list[str] | str,
    *,
    project_id: str,
    projects_root: Path | str,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    cpu_seconds: int | None = DEFAULT_CPU_SECONDS,
    memory_bytes: int | None = DEFAULT_MEMORY_BYTES,
    nofile: int | None = DEFAULT_NOFILE,
    env: dict[str, str] | None = None,
    extra_path_entries: list[str] | None = None,
    allow_shell: bool = False,
    allowed_executables: tuple[str, ...] | None = DEFAULT_ALLOWED_EXECUTABLES,
    allow_network: bool = False,
    require_network_isolation: bool = True,
    scrub_secrets: bool = True,
) -> ExecutionResult:
    """Run ``cmd`` in the project's sandbox with resource limits + timeout.

    Sandbox policy (2026-05-31 audit, item 4 — "no unrestricted shell or
    outbound network"):

      * **No shell.** A string ``cmd`` (which would run via ``shell=True``) is
        refused unless ``allow_shell=True`` is passed explicitly. The argv-list
        form is the only default path.
      * **Executable allowlist.** ``cmd[0]``'s basename must be in
        ``allowed_executables`` (default: python/python3/pytest). This blocks a
        generated ``["bash","-c",...]`` / ``["curl",...]`` from smuggling in a
        shell or a network fetch. Pass ``allowed_executables=None`` to disable
        the allowlist (escape hatch — use only for trusted callers).
      * **No outbound network.** With ``allow_network=False`` (default) the
        child is wrapped in an unprivileged user+network namespace (``unshare
        --map-root-user --net``) that has no interfaces. If that mechanism is
        unavailable and ``require_network_isolation`` is True, the call raises
        rather than silently running with network access.

    Args:
        cmd: argv list (preferred). A string is the shell form — refused unless
            ``allow_shell``.
        project_id: subdirectory under ``projects_root`` to sandbox in.
        projects_root: usually ``cfg.root / "projects"``.
        timeout_seconds: wall-clock kill switch (KICKOFF.md §10: 30 min default).
        cpu_seconds: RLIMIT_CPU cap.
        memory_bytes: RLIMIT_AS cap. macOS may silently ignore.
        nofile: RLIMIT_NOFILE cap.
        env: replaces ``os.environ`` for the child. If None, child inherits parent.
        extra_path_entries: prepended to ``$PATH``.
        allow_shell: permit a string ``cmd`` to run via the shell (default False).
        allowed_executables: argv[0] basename allowlist, or None to disable.
        allow_network: permit outbound network (default False → isolate).
        require_network_isolation: when network is disallowed but no isolation
            mechanism exists, raise instead of running networked (default True).
        scrub_secrets: when inheriting the parent env (``env is None``), strip
            secret-bearing vars (``*_API_KEY``/``*_TOKEN``/… and a name backstop)
            so untrusted generated code can't read them (default True). Ignored
            when ``env`` is supplied explicitly.
    """
    projects_root = Path(projects_root)
    sandbox = _ensure_sandbox(project_id, projects_root)
    log_dir = _ensure_log_dir(project_id, projects_root)

    is_posix = sys.platform != "win32"

    use_shell = isinstance(cmd, str)
    if use_shell and not allow_shell:
        raise SandboxPolicyError(
            "execute refused a shell-string command (arbitrary shell is "
            "disabled); pass an argv list, or allow_shell=True to override"
        )
    cmd_list = [cmd] if use_shell else list(cmd)
    if not cmd_list:
        raise SandboxPolicyError("execute received an empty command")

    # Executable allowlist (argv-list form only; a shell string is opaque).
    if not use_shell and allowed_executables is not None:
        exe = os.path.basename(str(cmd_list[0]))
        if exe not in allowed_executables:
            raise SandboxPolicyError(
                f"execute refused executable '{exe}': not in allowlist "
                f"{sorted(allowed_executables)} (blocks shells / curl / wget / nc). "
                f"Write a python script and run it, or pass allowed_executables."
            )

    # Outbound-network isolation. Wrap the argv list in an unprivileged netns.
    network_isolated = False
    if not allow_network:
        have_unshare = is_posix and shutil.which("unshare") is not None
        if have_unshare and not use_shell:
            cmd_list = [*_NETNS_WRAPPER, *cmd_list]
            network_isolated = True
        elif require_network_isolation and is_posix:
            raise NetworkIsolationUnavailable(
                "network isolation required but 'unshare' is unavailable "
                "(install util-linux) or cmd is a shell string; pass "
                "allow_network=True to run with network access knowingly"
            )
        else:
            # Windows-native (no namespaces) or isolation explicitly not
            # required: we cannot enforce it here — surface it loudly.
            log.warning(
                "execute.network_not_isolated",
                reason="no unshare / non-posix",
                platform=sys.platform,
            )

    # Inherit the parent environment only when the caller did not supply one,
    # and scrub secrets out of that inherited copy by default so untrusted
    # generated code can't read API keys / tokens. An explicitly-passed env is
    # the caller's responsibility and is used verbatim.
    if env is not None:
        child_env = dict(env)
    else:
        child_env = dict(os.environ)
        if scrub_secrets:
            child_env = _scrub_secret_env(child_env)
    if extra_path_entries:
        existing = child_env.get("PATH", "")
        child_env["PATH"] = os.pathsep.join([*extra_path_entries, existing]) if existing else os.pathsep.join(extra_path_entries)

    stdout_path = log_dir / "stdout.log"
    stderr_path = log_dir / "stderr.log"

    log.info(
        "execute.start",
        project_id=project_id,
        sandbox=str(sandbox),
        log_dir=str(log_dir),
        cmd=cmd_list[:6],
        timeout_seconds=timeout_seconds,
        network_isolated=network_isolated,
    )

    started = time.monotonic()
    timed_out = False
    proc: subprocess.Popen[bytes] | None = None
    preexec = (
        (lambda: _set_resource_limits(cpu_seconds, memory_bytes, nofile))
        if is_posix
        else None
    )

    try:
        with stdout_path.open("wb") as out_f, stderr_path.open("wb") as err_f:
            proc = subprocess.Popen(
                cmd_list,
                cwd=str(sandbox),
                stdout=out_f,
                stderr=err_f,
                env=child_env,
                preexec_fn=preexec,
                shell=use_shell,
                start_new_session=True,
            )
            try:
                exit_code = proc.wait(timeout=timeout_seconds)
            except subprocess.TimeoutExpired:
                timed_out = True
                if is_posix:
                    with contextlib.suppress(ProcessLookupError):
                        os.killpg(proc.pid, signal.SIGKILL)
                else:
                    proc.kill()
                # Bound the reap so a child stuck in uninterruptible sleep (or
                # re-parented outside the group) can't hang the call forever
                # while still holding the open log file handles.
                with contextlib.suppress(subprocess.TimeoutExpired):
                    proc.wait(timeout=30)
                exit_code = proc.returncode if proc.returncode is not None else -signal.SIGKILL
    finally:
        elapsed_ms = int((time.monotonic() - started) * 1000)

    stdout = stdout_path.read_text(encoding="utf-8", errors="replace")
    stderr = stderr_path.read_text(encoding="utf-8", errors="replace")

    log.info(
        "execute.done",
        project_id=project_id,
        exit_code=exit_code,
        timed_out=timed_out,
        duration_ms=elapsed_ms,
        stdout_chars=len(stdout),
        stderr_chars=len(stderr),
    )

    return ExecutionResult(
        exit_code=exit_code,
        stdout=stdout,
        stderr=stderr,
        duration_ms=elapsed_ms,
        timed_out=timed_out,
        sandbox_dir=str(sandbox),
        log_dir=str(log_dir),
        cmd=cmd_list,
        network_isolated=network_isolated,
    )


def reset_sandbox(project_id: str, projects_root: Path | str) -> Path:
    """Delete and recreate the sandbox for a project. Idempotent."""
    projects_root = Path(projects_root)
    sandbox = projects_root / project_id / "sandbox"
    if sandbox.exists():
        shutil.rmtree(sandbox)
    sandbox.mkdir(parents=True, exist_ok=True)
    return sandbox
