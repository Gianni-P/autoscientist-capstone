#!/usr/bin/env bash
# Back up the whole autoscientist project to a PRIVATE GitHub repo.
#
# One-command snapshot: stages everything (respecting .gitignore — secrets, the
# SQLite DB, the 100GB+ datasets, and run artifacts are all excluded), commits,
# ensures the private GitHub repo exists, and pushes. Safe to re-run any time.
#
# Auth: reads GITHUB_PERSONAL_ACCESS_TOKEN from .env (the same token the GitHub
# MCP publishing uses). The token is NEVER written to git config or printed.
#
#   bash scripts/backup_to_github.sh ["optional commit message"]
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

GH_USER="Gianni-P"
GH_REPO="autoscientist"

# --- load the token from .env ---
if [ -f .env ]; then set -a; . ./.env; set +a; fi
if [ -z "${GITHUB_PERSONAL_ACCESS_TOKEN:-}" ]; then
  echo "ERROR: GITHUB_PERSONAL_ACCESS_TOKEN not set (add it to .env)." >&2
  exit 1
fi

# Strip the token / any URL-embedded credential from anything we print.
redact() { sed -E "s#(https://[^:/]+:)[^@]+@#\1***@#g; s#${GITHUB_PERSONAL_ACCESS_TOKEN}#***#g"; }

# --- ensure a git identity exists ---
git config user.name  >/dev/null 2>&1 || git config user.name  "$GH_USER"
git config user.email >/dev/null 2>&1 || git config user.email "creepusdoesmc@gmail.com"

# --- stage + commit if there is anything new ---
git add -A
if git diff --cached --quiet; then
  echo "No changes to commit; pushing existing history."
else
  MSG="${1:-Backup $(date -u +%Y-%m-%dT%H:%MZ)}"
  git commit -q -m "$MSG"
  echo "Committed: $MSG"
fi

# --- ensure the private repo exists (idempotent) ---
code=$(curl -sS -o /dev/null -w '%{http_code}' \
  -H "Authorization: Bearer ${GITHUB_PERSONAL_ACCESS_TOKEN}" \
  -H "Accept: application/vnd.github+json" \
  "https://api.github.com/repos/${GH_USER}/${GH_REPO}")
if [ "$code" = "404" ]; then
  echo "Creating private repo ${GH_USER}/${GH_REPO} ..."
  curl -sS -X POST \
    -H "Authorization: Bearer ${GITHUB_PERSONAL_ACCESS_TOKEN}" \
    -H "Accept: application/vnd.github+json" \
    https://api.github.com/user/repos \
    -d "{\"name\":\"${GH_REPO}\",\"private\":true,\"description\":\"Multi-agent research pipeline: paper + supplementary + reproducible repo from research directions.\"}" \
    >/dev/null
elif [ "$code" != "200" ]; then
  echo "WARNING: unexpected status $code checking the repo; attempting push anyway." >&2
fi

# Keep a clean (token-free) origin for reference; push via an explicit
# credentialed URL so the token never lands in .git/config.
git remote get-url origin >/dev/null 2>&1 \
  || git remote add origin "https://github.com/${GH_USER}/${GH_REPO}.git"

git push "https://${GH_USER}:${GITHUB_PERSONAL_ACCESS_TOKEN}@github.com/${GH_USER}/${GH_REPO}.git" HEAD:main 2>&1 | redact
echo "Backup pushed to https://github.com/${GH_USER}/${GH_REPO} (private)."
