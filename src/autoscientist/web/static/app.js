/* autoscientist console — client.
 * One EventSource pushes change deltas; every action is an optimistic fetch.
 * No framework: small DOM helpers + targeted refreshers keep it instant. */
"use strict";
(() => {
  // ---- tiny helpers ----------------------------------------------------
  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => [...r.querySelectorAll(s)];
  const esc = (s) =>
    String(s ?? "").replace(/[&<>"']/g, (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

  const api = {
    async get(path) {
      const r = await fetch(path);
      if (!r.ok) throw new Error(`${r.status} ${path}`);
      return r.json();
    },
    async post(path, body) {
      const r = await fetch(path, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify(body || {}),
      });
      const data = await r.json().catch(() => ({}));
      return { ok: r.ok, data };
    },
  };

  const fmtCost = (v) => (v == null ? "" : "$" + Number(v).toFixed(4));
  const fmtUsd2 = (v) => "$" + Number(v || 0).toFixed(2);
  const parseTs = (s) => (s ? new Date(s) : null);
  const fmtClock = (s) => {
    const d = parseTs(s);
    if (!d || isNaN(d)) return "";
    return d.toTimeString().slice(0, 8);
  };
  const fmtRel = (s) => {
    const d = parseTs(s);
    if (!d || isNaN(d)) return "";
    const sec = Math.round((Date.now() - d.getTime()) / 1000);
    if (sec < 5) return "just now";
    if (sec < 60) return sec + "s ago";
    const m = Math.round(sec / 60);
    if (m < 60) return m + "m ago";
    const h = Math.round(m / 60);
    if (h < 24) return h + "h ago";
    return Math.round(h / 24) + "d ago";
  };

  const ROLE_ICON = { user: "📨", assistant: "🤖", tool: "🔧", handoff: "↪", system: "⚙" };

  // ---- state -----------------------------------------------------------
  const state = {
    runs: [],
    pending: [],
    selectedRunId: null,
    detail: null,
    tab: "activity",
    feed: { cursor: 0, atBottom: true, runId: null },
    openCpId: null,
    promptCache: {},
    models: null,   // model catalog for the per-leg picker (lazy-loaded)
  };

  // Selectable models / orchestrator info for the checkpoint model picker.
  async function ensureModels() {
    if (state.models) return state.models;
    try { state.models = await api.get("/api/models"); }
    catch (e) {
      state.models = { models: [], agent_defaults: {}, orchestratable: [], orchestrator: { available: false } };
    }
    return state.models;
  }

  // ---- toasts ----------------------------------------------------------
  function toast(msg, kind = "") {
    const t = document.createElement("div");
    t.className = "toast" + (kind ? " toast-" + kind : "");
    t.innerHTML = `<span class="t-icon">${kind === "error" ? "⚠️" : kind === "success" ? "✅" : "ℹ️"}</span><span>${esc(msg)}</span>`;
    $("#toasts").appendChild(t);
    setTimeout(() => { t.classList.add("out"); setTimeout(() => t.remove(), 240); }, 3200);
  }

  // ---- budget ----------------------------------------------------------
  function renderBudget(b) {
    if (!b) return;
    $("#budget-spent").textContent = fmtUsd2(b.spent);
    $("#budget-cap").textContent = fmtUsd2(b.cap);
    $("#budget-month").textContent = b.month || "";
    const fill = $("#budget-fill");
    fill.style.width = Math.min(100, (b.pct || 0) * 100) + "%";
    fill.classList.toggle("warn", b.pct >= 0.75 && b.pct < 0.92);
    fill.classList.toggle("danger", b.pct >= 0.92);
  }

  // ---- sidebar ---------------------------------------------------------
  function renderPending() {
    const sec = $("#pending-section");
    const list = $("#pending-list");
    $("#pending-count").textContent = state.pending.length;
    if (!state.pending.length) { sec.hidden = true; list.innerHTML = ""; return; }
    sec.hidden = false;
    list.innerHTML = "";
    state.pending.forEach((cp) => {
      const card = document.createElement("div");
      card.className = "pending-card";
      card.innerHTML =
        `<div class="pending-card-title">${esc(cp.title)}</div>
         <div class="pending-card-sub">${esc(cp.from_agent)} → ${esc(cp.to_agent)} · ${fmtRel(cp.created_at)}</div>
         ${cp.loop_cap ? `<div class="loop-flag">⚠ revision-loop cap hit</div>` : ""}`;
      card.onclick = () => { selectRun(cp.run_id); openCheckpoint(cp.checkpoint_id); };
      list.appendChild(card);
    });
  }

  function renderRuns() {
    const list = $("#runs-list");
    $("#runs-count").textContent = state.runs.length;
    list.innerHTML = "";
    state.runs.forEach((r) => {
      const card = document.createElement("div");
      card.className = "run-card s-" + r.status + (r.run_id === state.selectedRunId ? " selected" : "");
      card.innerHTML =
        `<div class="run-card-top">
           <span class="status-dot"></span>
           <span class="run-card-project">${esc(r.project_id)}</span>
         </div>
         <div class="run-card-agent">${r.current_agent ? "🤖 " + esc(r.current_agent) : '<span class="muted">—</span>'}</div>
         <div class="run-card-sub">
           <span>${esc(r.status)}</span><span>·</span>
           <span>${fmtRel(r.last_activity)}</span><span>·</span>
           <span class="mono">${fmtCost(r.total_cost)}</span>
         </div>`;
      card.onclick = () => selectRun(r.run_id);
      list.appendChild(card);
    });
  }

  // ---- run view --------------------------------------------------------
  async function selectRun(runId) {
    if (state.selectedRunId === runId && state.detail) { closeNav(); return; }
    state.selectedRunId = runId;
    renderRuns();
    closeNav();
    $("#empty-state").hidden = true;
    $("#run-view").hidden = false;
    await refreshDetail();
    state.tab = "activity";
    syncTabs();
    await loadFeed(true);
  }

  async function refreshDetail() {
    if (!state.selectedRunId) return;
    try {
      state.detail = await api.get(`/api/runs/${state.selectedRunId}`);
    } catch (e) { return; }
    renderHero();
    renderStepper();
    renderNow();
    if (state.tab === "handoffs") loadTimeline();
    if (state.tab === "checkpoints") loadCheckpointsHistory();
    if (state.tab === "agents") renderAgents();
  }

  function statusPill(status) {
    return `<span class="status-pill pill-${esc(status)}">${esc(status)}</span>`;
  }

  function renderHero() {
    const d = state.detail; if (!d) return;
    const run = d.run;
    $("#run-status").outerHTML = `<span class="status-pill pill-${esc(run.status)}" id="run-status">${esc(run.status)}</span>`;
    $("#run-project").textContent = run.project_id;
    $("#run-meta").innerHTML = [
      `<span><span class="k">run</span> <span class="mono">${esc(run.run_id)}</span></span>`,
      `<span><span class="k">started</span> ${fmtRel(run.started_at)}</span>`,
      d.totals ? `<span><span class="k">events</span> ${d.totals.messages}</span>` : "",
      d.totals ? `<span><span class="k">tokens</span> ${(d.totals.prompt_tokens + d.totals.completion_tokens).toLocaleString()}</span>` : "",
      d.totals ? `<span><span class="k">cost</span> <span class="mono">${fmtCost(d.totals.cost)}</span></span>` : "",
      run.note ? `<span><span class="k">note</span> ${esc(run.note)}</span>` : "",
    ].filter(Boolean).join("");
    renderControls();
  }

  function renderControls() {
    const d = state.detail; if (!d) return;
    const host = $("#run-controls");
    host.innerHTML = "";
    const run = d.run, pause = d.pause, runId = run.run_id;
    const mkBtn = (label, cls, fn, disabled) => {
      const b = document.createElement("button");
      b.className = "btn " + cls; b.textContent = label; b.disabled = !!disabled;
      if (fn) b.onclick = fn;
      host.appendChild(b);
    };
    if (run.status === "running") {
      if (pause && pause.pause_requested && !pause.paused_at) {
        mkBtn("⏸ Pausing…", "", null, true);
        mkBtn("Cancel pause", "btn-ghost", () => act(`/api/runs/${runId}/cancel-pause`, "Pause cancelled"));
      } else {
        mkBtn("⏸ Pause", "", () => act(`/api/runs/${runId}/pause`, "Pause requested — stops at next agent boundary"));
      }
    } else if (run.status === "paused") {
      if (d.pending_checkpoint) {
        const b = document.createElement("span");
        b.className = "small muted"; b.textContent = "Paused at a checkpoint — open it to resume.";
        host.appendChild(b);
      } else if (pause && pause.is_active) {
        mkBtn("▶ Resume", "btn-primary", () => act(`/api/runs/${runId}/resume`, "Resuming in background"));
      }
    }
  }

  async function act(path, okMsg) {
    const { ok, data } = await api.post(path);
    if (ok && data.ok !== false) { toast(okMsg, "success"); setTimeout(refreshDetail, 300); }
    else toast((data && data.error) || "Action failed", "error");
  }

  function renderStepper() {
    const d = state.detail; if (!d) return;
    const host = $("#stepper");
    host.innerHTML = "";
    d.stages.forEach((s) => {
      const stepCls =
        s.status === "approved" ? "done" :
        s.status === "modified" ? "modified" :
        s.status === "rejected" ? "rejected" :
        s.status === "pending" ? "current" : "";
      const node =
        s.status === "approved" ? "✓" :
        s.status === "modified" ? "±" :
        s.status === "rejected" ? "✕" : String(s.stage);
      const div = document.createElement("div");
      div.className = "step " + stepCls + (s.checkpoint_id ? " clickable" : "");
      div.innerHTML =
        `<div class="connector"></div>
         <div class="node">${node}</div>
         <div class="label">${esc(s.title.replace(/^Stage \d+ — /, ""))}</div>`;
      if (s.checkpoint_id) div.onclick = () => openCheckpoint(s.checkpoint_id);
      host.appendChild(div);
    });
  }

  function renderNow() {
    const d = state.detail; if (!d) return;
    const host = $("#now-card");
    if (d.pending_checkpoint) {
      const cp = d.pending_checkpoint;
      host.className = "now-card action";
      host.innerHTML =
        `<div class="now-row">
           <div class="now-icon">⏳</div>
           <div class="now-meta">
             <div class="now-label">Action needed · ${esc(cp.title)}</div>
             <div class="now-agent">Awaiting your decision</div>
             <div class="now-detail">${esc(cp.from_agent)} → ${esc(cp.to_agent)}${cp.loop_cap ? " · ⚠ loop cap" : ""}</div>
           </div>
         </div>`;
      const b = document.createElement("button");
      b.className = "btn btn-primary"; b.textContent = "Open checkpoint →";
      b.onclick = () => openCheckpoint(cp.checkpoint_id);
      host.appendChild(b);
      return;
    }
    host.className = "now-card";
    const running = d.run.status === "running";
    const last = d.last_event;
    host.innerHTML =
      `<div class="now-row">
         <div class="now-icon">${running ? '<div class="now-spin"></div>' : "🧪"}</div>
         <div class="now-meta">
           <div class="now-label">${running ? "Now running" : "Run " + esc(d.run.status)}</div>
           <div class="now-agent">${d.current_agent ? esc(d.current_agent) : "—"}</div>
           <div class="now-detail">${last ? esc(last.role) + " · " + esc(last.preview || "") : '<span class="muted">no activity yet</span>'}</div>
         </div>
       </div>`;
  }

  // ---- tabs ------------------------------------------------------------
  function syncTabs() {
    $$("#tabs .seg").forEach((b) => b.classList.toggle("active", b.dataset.tab === state.tab));
    $("#panel-activity").hidden = state.tab !== "activity";
    $("#panel-handoffs").hidden = state.tab !== "handoffs";
    $("#panel-checkpoints").hidden = state.tab !== "checkpoints";
    $("#panel-agents").hidden = state.tab !== "agents";
  }
  $("#tabs").addEventListener("click", (e) => {
    const seg = e.target.closest(".seg"); if (!seg) return;
    state.tab = seg.dataset.tab;
    syncTabs();
    if (state.tab === "handoffs") loadTimeline();
    if (state.tab === "checkpoints") loadCheckpointsHistory();
    if (state.tab === "agents") renderAgents();
    if (state.tab === "activity") scrollFeedBottom();
  });

  // ---- activity feed ---------------------------------------------------
  const feedEl = () => $("#feed");
  function eventRow(ev) {
    const row = document.createElement("div");
    const detailable = ev.role === "tool" || ev.role === "handoff" || (ev.content && ev.content.length > 200);
    row.className = `evt role-${ev.role}` + (detailable ? " has-detail" : "");
    let tag = "";
    if (ev.role === "assistant") tag = `<span class="evt-tag">${esc(ev.model || "?")}</span>` +
      (ev.cost != null ? ` <span class="evt-cost">${fmtCost(ev.cost)}${ev.cache_hit ? " · cached" : ""}</span>` : "");
    else if (ev.role === "tool" && ev.tool) tag = `<span class="evt-tag ${ev.tool.ok ? "ok" : "err"}">${esc(ev.tool.name || "?")}</span>` +
      (ev.tool.duration_ms != null ? ` <span class="evt-cost">${ev.tool.duration_ms}ms</span>` : "");
    else if (ev.role === "handoff" && ev.handoff) tag = `<span class="evt-tag">${esc(ev.handoff.decision || "?")} → ${esc(ev.handoff.next_agent || "?")}</span>`;
    row.innerHTML =
      `<div class="evt-time">${fmtClock(ev.at)}</div>
       <div class="evt-icon">${ROLE_ICON[ev.role] || "·"}</div>
       <div class="evt-main">
         <div class="evt-head"><span class="evt-agent">${esc(ev.agent)}</span>${tag}</div>
         ${ev.preview ? `<div class="evt-preview">${esc(ev.preview)}</div>` : ""}
       </div>`;
    if (detailable) row.onclick = () => showEventDetail(ev);
    return row;
  }

  async function loadFeed(reset) {
    const runId = state.selectedRunId;
    if (reset) { state.feed = { cursor: 0, atBottom: true, runId }; feedEl().innerHTML = ""; }
    const data = await api.get(`/api/runs/${runId}/messages?after=${state.feed.cursor}`);
    const frag = document.createDocumentFragment();
    data.events.forEach((ev) => {
      const r = eventRow(ev);
      if (!reset) r.classList.add("evt-appear");
      frag.appendChild(r);
    });
    feedEl().appendChild(frag);
    state.feed.cursor = data.cursor;
    if (state.feed.atBottom) scrollFeedBottom();
    else $("#jump-latest").hidden = false;
  }

  function scrollFeedBottom() {
    const f = feedEl();
    f.scrollTop = f.scrollHeight;
    state.feed.atBottom = true;
    $("#jump-latest").hidden = true;
  }
  feedEl().addEventListener("scroll", () => {
    const f = feedEl();
    state.feed.atBottom = f.scrollHeight - f.scrollTop - f.clientHeight < 40;
    if (state.feed.atBottom) $("#jump-latest").hidden = true;
  });
  $("#jump-latest").onclick = scrollFeedBottom;

  async function showEventDetail(ev) {
    let full = ev;
    if (ev.truncated) { try { full = await api.get(`/api/messages/${ev.id}`); } catch (_) {} }
    let body = "";
    if (ev.role === "tool" && ev.tool) {
      body += `<div class="kv"><span class="k">tool</span><span class="mono">${esc(ev.tool.name)}</span>`;
      body += `<span class="k">status</span><span>${ev.tool.ok ? "✓ ok" : "✕ error"}</span></div>`;
      if (ev.tool.input != null) body += `<div class="section-title">input</div><pre class="code">${esc(JSON.stringify(ev.tool.input, null, 2))}</pre>`;
      if (ev.tool.error) body += `<div class="banner banner-warn">${esc(ev.tool.error)}</div>`;
    }
    body += `<div class="section-title">${esc(ev.role)} content</div><pre class="code">${esc(full.content || "(empty)")}</pre>`;
    openModal(`${ROLE_ICON[ev.role] || ""} ${ev.agent}`, ev.role, body);
  }

  // ---- handoffs & prompts timeline ------------------------------------
  async function loadTimeline() {
    const panel = $("#panel-handoffs");
    panel.innerHTML = `<div class="skeleton" style="height:120px"></div>`;
    let data;
    try { data = await api.get(`/api/runs/${state.selectedRunId}/timeline`); }
    catch (e) { panel.innerHTML = `<div class="muted">Failed to load timeline.</div>`; return; }
    if (!data.activations.length) { panel.innerHTML = `<div class="muted small">No agent activity recorded yet.</div>`; return; }
    panel.innerHTML = "";
    const wrap = document.createElement("div");
    wrap.className = "timeline";
    data.activations.forEach((a) => wrap.appendChild(activationCard(a)));
    panel.appendChild(wrap);
  }

  function block(title, contentHtml, opts = {}) {
    const open = opts.open ? " open" : "";
    return `<div class="block${open}">
      <div class="block-head"><span>${esc(title)}</span><span class="chev">›</span></div>
      <div class="block-body">${contentHtml}</div>
    </div>`;
  }

  function activationCard(a) {
    const card = document.createElement("div");
    card.className = "act-card";
    const tools = (a.tool_calls || []).map((t) =>
      `<span class="chip ${t.ok ? "ok" : "err"}">${esc(t.name)}</span>`).join("");
    let body = "";
    // The prompt for this agent = static system prompt + dynamic inbound.
    const promptInner =
      `<div class="form-actions" style="margin-bottom:8px">
         <button class="btn btn-sm" data-prompt="${esc(a.agent)}">View ${esc(a.agent)} system prompt</button>
       </div>
       <div class="section-title">Inbound payload (the prompt delivered to ${esc(a.agent)})</div>
       <pre class="code">${esc(a.inbound_prompt || "(no recorded inbound — resumed mid-chain)")}</pre>`;
    body += block("📨 Prompt for " + a.agent, promptInner, { open: false });
    if (a.output)
      body += block("🤖 Final output (" + a.assistant_turns + " turn" + (a.assistant_turns === 1 ? "" : "s") + ")",
        `<pre class="code">${esc(a.output)}</pre>`);
    let handoffBar = "";
    if (a.handoff && (a.handoff.decision || a.handoff.next_agent)) {
      const dec = a.handoff.decision || "";
      handoffBar = `<div class="handoff-bar">
        <span class="arrow">↪ handoff</span>
        ${dec ? `<span class="decision-tag decision-${esc(dec)}">${esc(dec)}</span>` : ""}
        <span class="act-spacer"></span>
        <span class="to">${esc(a.handoff.next_agent || "DONE")}</span>
      </div>`;
    }
    card.innerHTML =
      `<div class="act-head">
         <div class="act-seq">${a.seq}</div>
         <div>
           <div class="act-agent">${esc(a.agent)}</div>
           <div class="act-sub">${fmtClock(a.started_at)} · ${a.tokens.toLocaleString()} tok · ${fmtCost(a.cost)}</div>
         </div>
         <div class="act-spacer"></div>
         <div class="act-chips">${tools}</div>
       </div>
       <div class="act-body">${body}</div>
       ${handoffBar}`;
    // wire block toggles + prompt buttons
    $$(".block-head", card).forEach((h) => h.onclick = () => h.parentElement.classList.toggle("open"));
    $$("[data-prompt]", card).forEach((b) => b.onclick = (e) => { e.stopPropagation(); showSystemPrompt(b.dataset.prompt); });
    return card;
  }

  async function showSystemPrompt(agent) {
    let data = state.promptCache[agent];
    if (!data) {
      try { data = await api.get(`/api/agents/${agent}/prompt`); state.promptCache[agent] = data; }
      catch (e) { toast("No system prompt file for " + agent, "error"); return; }
    }
    openModal("⚙ System prompt", agent, `<pre class="code">${esc(data.system_prompt)}</pre>`);
  }

  // ---- agents tab ------------------------------------------------------
  function renderAgents() {
    const d = state.detail; if (!d) return;
    const panel = $("#panel-agents");
    if (!d.agents || !d.agents.length) { panel.innerHTML = `<div class="muted small">No agents yet.</div>`; return; }
    panel.innerHTML = `<div class="agent-grid">${d.agents.map((a) =>
      `<div class="agent-tile" data-agent="${esc(a.agent)}">
         <div class="name">${esc(a.agent)}</div>
         <div class="stat">${a.events} events · ${fmtCost(a.cost)}</div>
         <div class="stat">last ${fmtRel(a.last_at)}</div>
       </div>`).join("")}</div>`;
    $$(".agent-tile", panel).forEach((t) => t.onclick = () => showSystemPrompt(t.dataset.agent));
  }

  // ---- checkpoints history (go back to previous checkpoints) -----------
  async function loadCheckpointsHistory() {
    const panel = $("#panel-checkpoints");
    panel.innerHTML = `<div class="skeleton" style="height:72px"></div>`;
    let data;
    try { data = await api.get(`/api/runs/${state.selectedRunId}/checkpoints`); }
    catch (e) { panel.innerHTML = `<div class="muted small">Failed to load checkpoints.</div>`; return; }
    if (!data.checkpoints.length) { panel.innerHTML = `<div class="muted small">No checkpoints opened yet for this run.</div>`; return; }
    panel.innerHTML = `<div class="cp-history">${data.checkpoints.map(cpHistoryRow).join("")}</div>`;
    $$("#panel-checkpoints .cp-row").forEach((r) => r.onclick = () => openCheckpoint(r.dataset.cp));
  }

  function cpHistoryRow(cp) {
    const label = cp.decision === "rerun" ? "rerun" : cp.status;
    const pillCls = cp.decision === "rerun" ? "pill-modified" : "pill-" + cp.status;
    return `<div class="cp-row" data-cp="${esc(cp.checkpoint_id)}">
      <span class="cp-stage-badge">${cp.stage}</span>
      <div class="cp-row-main">
        <div class="cp-row-title">${esc(cp.title.replace(/^Stage \d+ — /, ""))}${cp.loop_cap ? ' <span class="loop-flag">⚠ loop cap</span>' : ""}</div>
        <div class="cp-row-sub">${esc(cp.from_agent)} → ${esc(cp.to_agent)} · opened ${fmtRel(cp.created_at)}${cp.resolved_at ? " · resolved " + fmtRel(cp.resolved_at) : ""}</div>
      </div>
      <span class="status-pill ${pillCls}">${esc(label)}</span>
    </div>`;
  }

  // ---- modal -----------------------------------------------------------
  function openModal(title, status, bodyHtml) {
    $("#cp-title").textContent = title;
    $("#cp-status").outerHTML = status
      ? `<span class="status-pill pill-${esc(status)}" id="cp-status">${esc(status)}</span>`
      : `<span class="status-pill" id="cp-status" style="display:none"></span>`;
    $("#modal-body").innerHTML = bodyHtml;
    $("#modal-backdrop").hidden = false;
  }
  function closeModal() { $("#modal-backdrop").hidden = true; state.openCpId = null; }
  $("#modal-close").onclick = closeModal;
  $("#modal-backdrop").addEventListener("click", (e) => { if (e.target === $("#modal-backdrop")) closeModal(); });
  document.addEventListener("keydown", (e) => { if (e.key === "Escape" && !$("#modal-backdrop").hidden) closeModal(); });

  // ---- checkpoint resolver --------------------------------------------
  async function openCheckpoint(cpId) {
    state.openCpId = cpId;
    openModal("Checkpoint", "", `<div class="skeleton" style="height:200px"></div>`);
    let cp;
    try { cp = await api.get(`/api/checkpoints/${cpId}`); }
    catch (e) { $("#modal-body").innerHTML = `<div class="banner banner-warn">Checkpoint not found.</div>`; return; }
    await ensureModels();
    renderCheckpoint(cp);
  }

  function renderCheckpoint(cp) {
    $("#cp-title").textContent = cp.title;
    $("#cp-status").outerHTML = `<span class="status-pill pill-${esc(cp.status)}" id="cp-status">${esc(cp.status)}</span>`;
    const parts = [];
    parts.push(`<div class="cp-meta">
      <span><span class="k">from</span> <span class="mono">${esc(cp.from_agent)}</span> → <span class="mono">${esc(cp.to_agent)}</span></span>
      <span><span class="k">opened</span> ${fmtRel(cp.created_at)}</span>
      <span class="mono">${esc(cp.checkpoint_id)}</span>
    </div>`);
    if (cp.extra && cp.extra.loop_cap_exceeded) {
      parts.push(`<div class="banner banner-warn"><b>Revision loop cap exceeded.</b> code_review fired ${esc(cp.extra.cycles)}× (cap ${esc(cp.extra.max_cycles)}). Next step is back to code_gen. Approve to retry once more, modify to give explicit instructions, or reject to cancel.</div>`);
    }
    if (cp.summary) parts.push(`<div class="banner banner-info">${esc(cp.summary)}</div>`);

    // stage-specific payload
    parts.push(`<div>${renderStage(cp)}</div>`);

    // raw / default payload (collapsible)
    parts.push(block("Raw agent output", `<pre class="code">${esc(cp.raw || "(empty)")}</pre>`));
    parts.push(block("Default next-agent payload", `<pre class="code">${esc(cp.default_payload || "(empty)")}</pre>`));

    // decision form OR resolution summary
    if (cp.status === "pending") parts.push(decisionForm(cp));
    else parts.push(resolutionSummary(cp));

    // Q&A
    parts.push(`<div><div class="section-title">Ask the orchestrator</div><div id="qa-thread" class="qa-thread"></div>
      <div class="field" style="margin-top:8px">
        <textarea id="qa-input" placeholder="Question (persists; does not resolve the checkpoint)"></textarea>
        <div class="form-actions"><button class="btn" id="qa-send">Add question</button></div>
      </div></div>`);

    $("#modal-body").innerHTML = parts.join("");
    wireCheckpoint(cp);
  }

  function wireCheckpoint(cp) {
    $$("#modal-body .block-head").forEach((h) => h.onclick = () => h.parentElement.classList.toggle("open"));
    renderQA(cp.questions);
    const send = $("#qa-send");
    if (send) send.onclick = async () => {
      const v = $("#qa-input").value.trim(); if (!v) return;
      send.disabled = true;
      const { ok, data } = await api.post(`/api/checkpoints/${cp.checkpoint_id}/questions`, { content: v });
      send.disabled = false;
      if (ok && data.ok) { $("#qa-input").value = ""; const fresh = await api.get(`/api/checkpoints/${cp.checkpoint_id}`); renderQA(fresh.questions); }
      else toast("Could not add question", "error");
    };
    if (cp.status === "pending") wireDecisionForm(cp);
  }

  function renderQA(questions) {
    const host = $("#qa-thread"); if (!host) return;
    if (!questions || !questions.length) { host.innerHTML = `<div class="muted small">No questions yet.</div>`; return; }
    host.innerHTML = questions.map((q) =>
      `<div class="qa-msg qa-${q.role === "operator" ? "operator" : "assistant"}">
         ${esc(q.content)}
         <div class="qa-meta">${fmtRel(q.at)}${q.agent_used ? " · via " + esc(q.agent_used) : ""}${q.cost != null ? " · " + fmtCost(q.cost) : ""}</div>
       </div>`).join("");
  }

  function decisionForm(cp) {
    return `<div class="decision-form" id="decision-form">
      <div class="section-title">Decide</div>
      <div class="decision-choices wide" id="decision-choices">
        <div class="choice" data-d="approve" aria-pressed="true">✓ Approve</div>
        <div class="choice" data-d="modify" aria-pressed="false">± Approve w/ changes</div>
        <div class="choice" data-d="rerun" aria-pressed="false">↻ Re-run w/ nudge</div>
        <div class="choice" data-d="reject" aria-pressed="false">✕ Reject</div>
      </div>
      <div id="decision-detail"></div>
      <div id="model-picker" class="model-picker"></div>
      <div class="form-actions">
        <button class="btn btn-primary" id="d-submit">Approve &amp; resume run</button>
        <span class="muted small" id="d-hint"></span>
      </div>
    </div>`;
  }

  function wireDecisionForm(cp) {
    let decision = "approve";
    let modifyMode = "describe";       // "describe" | "edit"
    const toA = cp.to_agent, fromA = cp.from_agent;
    const detail = $("#decision-detail");
    const submit = $("#d-submit");
    const hint = $("#d-hint");

    // ---- per-leg model picker -----------------------------------------
    const cat = state.models || { models: [], agent_defaults: {}, orchestratable: [], orchestrator: { available: false } };
    const modelSel = {};   // agent -> chosen alias ("" = use config default)

    // Which agents a decision will actually run (so the picker is scoped to
    // exactly the agents this approval drives): approve/modify → the next leg;
    // rerun → the agent that produced this checkpoint; reject → none.
    function agentsForDecision(d) {
      if (d === "rerun") return fromA ? [fromA] : [];
      if (d === "approve" || d === "modify") return cp.next_leg_agents || [];
      return [];
    }
    function modelOptions(agent) {
      const def = cat.agent_defaults[agent];
      const defM = (cat.models.find((m) => m.alias === def) || {}).model_id || def || "config default";
      let opts = `<option value="">default · ${esc(defM)}</option>`;
      if ((cat.orchestratable || []).includes(agent) && cat.orchestrator && cat.orchestrator.available)
        opts += `<option value="${esc(cat.orchestrator.value)}">${esc(cat.orchestrator.label)}</option>`;
      cat.models.forEach((m) => {
        opts += `<option value="${esc(m.alias)}">${esc(m.model_id)} · ${esc(m.tier)} · ${esc(m.price)}</option>`;
      });
      return opts;
    }
    function renderModelPicker(d) {
      const host = $("#model-picker"); if (!host) return;
      const agents = agentsForDecision(d);
      if (!agents.length || !cat.models.length) { host.innerHTML = ""; return; }
      const title = d === "rerun"
        ? `Model for ${esc(fromA)} <span class="muted">(optional)</span>`
        : `Models for this leg <span class="muted">(optional · resets after the next checkpoint)</span>`;
      host.innerHTML =
        `<div class="section-title">${title}</div>
         <div class="model-rows">${agents.map((a) =>
           `<div class="model-row">
              <span class="model-agent">${esc(a)}</span>
              <select class="model-select" data-agent="${esc(a)}">${modelOptions(a)}</select>
            </div>`).join("")}</div>`;
      $$("#model-picker .model-select").forEach((sel) => {
        sel.value = modelSel[sel.dataset.agent] || "";
        sel.onchange = () => { modelSel[sel.dataset.agent] = sel.value; };
      });
    }

    function renderDetail() {
      if (decision === "approve") {
        detail.innerHTML = "";
        hint.textContent = `Forwards the default payload to ${toA} and resumes.`;
        submit.textContent = "Approve & resume"; submit.className = "btn btn-primary";
      } else if (decision === "modify") {
        const prevInstr = $("#d-instructions")?.value;
        const prevPayload = $("#d-payload")?.value;
        detail.innerHTML =
          `<div class="submode" id="modify-submode">
             <button type="button" class="submode-btn ${modifyMode === "describe" ? "active" : ""}" data-m="describe">Describe a change</button>
             <button type="button" class="submode-btn ${modifyMode === "edit" ? "active" : ""}" data-m="edit">Edit handoff prompt</button>
           </div>
           <div class="field" ${modifyMode !== "describe" ? "hidden" : ""}>
             <label>Describe the change for ${esc(toA)}</label>
             <textarea id="d-instructions" placeholder="e.g. use 5 seeds not 3; drop dataset Y">${esc(prevInstr || "")}</textarea>
           </div>
           <div class="field" ${modifyMode !== "edit" ? "hidden" : ""}>
             <label>Handoff prompt delivered to ${esc(toA)} — edit directly</label>
             <textarea id="d-payload" class="mono-area">${esc(prevPayload != null ? prevPayload : (cp.default_payload || ""))}</textarea>
           </div>`;
        $$("#modify-submode .submode-btn").forEach((b) => b.onclick = () => { modifyMode = b.dataset.m; renderDetail(); });
        hint.textContent = modifyMode === "describe"
          ? `Appends your instructions to the payload for ${toA}.`
          : `Replaces the payload sent to ${toA}.`;
        submit.textContent = "Approve with changes & resume"; submit.className = "btn btn-primary";
      } else if (decision === "rerun") {
        const prevNudge = $("#d-nudge")?.value;
        detail.innerHTML =
          `<div class="field">
             <label>Nudge for <b>${esc(fromA)}</b> — re-runs it on its original prompt plus this nudge</label>
             <textarea id="d-nudge" placeholder="What should ${esc(fromA)} do differently this time?">${esc(prevNudge || "")}</textarea>
           </div>`;
        hint.textContent = `Re-runs ${fromA}, then pauses again at this checkpoint.`;
        submit.textContent = `↻ Re-run ${fromA}`; submit.className = "btn btn-primary";
      } else {  // reject
        detail.innerHTML =
          `<div class="field">
             <label>Reason <span class="muted">(optional)</span></label>
             <textarea id="d-reason" placeholder="Why stop here?"></textarea>
           </div>`;
        hint.textContent = "Stops the run here (cancelled). It will not resume.";
        submit.textContent = "Reject & stop run"; submit.className = "btn btn-danger";
      }
      renderModelPicker(decision);
    }

    $$("#decision-choices .choice").forEach((c) => c.onclick = () => {
      decision = c.dataset.d;
      $$("#decision-choices .choice").forEach((x) => x.setAttribute("aria-pressed", x === c));
      renderDetail();
    });
    renderDetail();

    submit.onclick = async () => {
      submit.disabled = true;
      let body;
      if (decision === "approve") body = { decision: "approve" };
      else if (decision === "modify") body = modifyMode === "edit"
        ? { decision: "modify", modified_payload: $("#d-payload").value || null }
        : { decision: "modify", instructions: $("#d-instructions").value || null };
      else if (decision === "rerun") body = { decision: "rerun", instructions: $("#d-nudge").value || null };
      else body = { decision: "reject", instructions: $("#d-reason").value || null };

      // Attach the operator's per-leg model picks (scoped to the agents this
      // decision actually runs). Empty selections = keep the config default.
      const relevant = new Set(agentsForDecision(decision));
      const overrides = {};
      Object.keys(modelSel).forEach((a) => { if (modelSel[a] && relevant.has(a)) overrides[a] = modelSel[a]; });
      if (Object.keys(overrides).length) body.model_overrides = overrides;

      const { ok, data } = await api.post(`/api/checkpoints/${cp.checkpoint_id}/resolve`, body);
      submit.disabled = false;
      if (ok && data.ok) {
        const msg = decision === "reject" ? "Rejected — run stopped"
          : decision === "rerun" ? `Re-running ${fromA} in the background…`
          : "Resolved — resuming in the background";
        toast(msg, "success");
        closeModal();
        setTimeout(() => { refreshOverview(); refreshDetail(); }, 400);
      } else {
        toast((data && data.error) || "Action failed", "error");
      }
    };
  }

  function resolutionSummary(cp) {
    const op = cp.operator_input || {};
    let html = `<div class="banner banner-info">Resolved as <b>${esc(cp.status)}</b>${cp.resolved_at ? " · " + fmtRel(cp.resolved_at) : ""} (decision: ${esc(op.decision || "?")}).</div>`;
    if (op.instructions) html += `<div class="section-title">Operator instructions</div><pre class="code">${esc(op.instructions)}</pre>`;
    if (op.modified_payload) html += block("Operator-overridden payload", `<pre class="code">${esc(op.modified_payload)}</pre>`);
    if (op.model_overrides && Object.keys(op.model_overrides).length) {
      const rows = Object.entries(op.model_overrides)
        .map(([a, m]) => `<span class="model-chip"><b>${esc(a)}</b> → ${esc(m)}</span>`).join("");
      html += `<div class="section-title">Model overrides (this leg)</div><div class="model-chips">${rows}</div>`;
    }
    return html;
  }

  // ---- stage-specific payload renderers (mirror the Streamlit layouts) --
  function renderStage(cp) {
    const p = cp.parsed;
    if (!p) return jsonFallback(cp.raw);
    try {
      switch (cp.stage) {
        case 1: return stageIdea(p);
        case 2: return stageMethod(p);
        case 3: return stagePrelim(p);
        case 4: return `<div class="section-title">Full results validation</div>` + validatorBody(p);
        case 5: return stageDraft(p);
        default: return jsonBlock(p);
      }
    } catch (e) { return jsonBlock(p); }
  }

  const jsonBlock = (o) => `<pre class="code">${esc(JSON.stringify(o, null, 2))}</pre>`;
  const jsonFallback = (raw) => `<div class="banner banner-warn">Agent output did not parse as JSON — showing raw text.</div><pre class="code">${esc(raw || "(empty)")}</pre>`;

  function table(rows) {
    if (!rows || !rows.length) return "";
    const cols = [...new Set(rows.flatMap((r) => Object.keys(r || {})))];
    return `<table class="dtable"><thead><tr>${cols.map((c) => `<th>${esc(c)}</th>`).join("")}</tr></thead>
      <tbody>${rows.map((r) => `<tr>${cols.map((c) => `<td>${esc(typeof r[c] === "object" ? JSON.stringify(r[c]) : r[c] ?? "")}</td>`).join("")}</tr>`).join("")}</tbody></table>`;
  }
  const ul = (items) => items && items.length ? `<ul class="list-tight">${items.map((i) => `<li>${esc(typeof i === "object" ? JSON.stringify(i) : i)}</li>`).join("")}</ul>` : "";

  function stageIdea(p) {
    let h = `<div class="kv"><span class="k">top pick</span><span>idea #${esc(p.top_pick)}</span>
      <span class="k">ranked</span><span class="mono">${esc(JSON.stringify(p.ranked_indices || []))}</span></div>`;
    if ((p.operator_questions || []).length) h += `<div class="banner banner-info"><b>Critic asks you:</b>${ul(p.operator_questions)}</div>`;
    (p.critiques || []).forEach((c) => {
      const rec = (c.recommendation || "?").toUpperCase();
      h += block(`Idea ${c.idea_index ?? "?"} — ${rec}`,
        `${ul((c.concerns || []).map((x) => "⚠ " + x))}
         ${c.kill_criteria ? `<div class="section-title">Kill criteria</div>${ul(c.kill_criteria)}` : ""}
         ${c.potential_confounds ? `<div class="small"><b>Confounds:</b> ${esc((c.potential_confounds || []).join(", "))}</div>` : ""}
         ${c.rationale ? `<div class="small muted">${esc(c.rationale)}</div>` : ""}`,
        { open: c.idea_index === p.top_pick });
    });
    return h;
  }

  function stageMethod(p) {
    const plan = p.plan || p;
    let h = `<div class="kv"><span class="k">research question</span><span>${esc(plan.research_question || "?")}</span></div>`;
    if ((plan.hypotheses || []).length) h += `<div class="section-title">Hypotheses</div>` +
      ul(plan.hypotheses.map((x) => `${x.id || "?"}: ${x.statement || "?"} (dir: ${x.predicted_direction || "?"})`));
    [["Datasets", plan.datasets], ["Baselines", plan.baselines], ["Metrics", plan.metrics], ["Experiments", plan.experiments]].forEach(([t, rows]) => {
      if (rows && rows.length) h += `<div class="section-title">${t}</div>${table(rows)}`;
    });
    const sp = plan.stats_plan || {};
    if (Object.keys(sp).length) h += `<div class="small"><b>Stats:</b> ${esc(sp.primary_test || "?")} · α=${esc(sp.alpha ?? "?")} · MC=${esc(sp.multiple_comparisons ?? "?")} · floor=${esc(sp.effect_size_floor ?? "?")}</div>`;
    if ((plan.pitfall_acks || []).length) h += `<div class="section-title">Pitfall acknowledgements</div>` +
      ul(plan.pitfall_acks.map((x) => `${x.pitfall || "?"} → ${x.mitigation || "?"}`));
    const sc = plan.stop_conditions || {};
    if (Object.keys(sc).length) h += `<div class="small"><b>Stop:</b> success → ${esc(sc.early_success || "?")}; abort → ${esc(sc.early_abort || "?")}</div>`;
    return h;
  }

  function stagePrelim(p) {
    const isReview = ("findings" in p || "verdict" in p) && !("checks" in p);
    if (isReview) {
      const v = p.verdict || "?";
      const vlabel = { pass: "✅ pass — advance to results_validator", revise: "🔁 revise — another code_gen pass", block: "⛔ block — halted on a blocker" }[v] || ("verdict: " + v);
      let h = `<div class="kv"><span class="k">verdict</span><span>${esc(vlabel)}</span></div>`;
      if (p.summary) h += `<div class="banner banner-info">${esc(p.summary)}</div>`;
      const rows = (p.findings || []).map((f) => ({
        severity: f.severity, category: f.category, file: f.file, lines: f.lines, issue: f.issue, fix: f.fix_suggestion,
      }));
      h += rows.length ? `<div class="section-title">Findings</div>${table(rows)}` : `<div class="small muted">No findings recorded.</div>`;
      return h;
    }
    return `<div class="banner banner-info">Subset run results — sanity-check before committing GPU-hours.</div>` + validatorBody(p);
  }

  function validatorBody(p) {
    let h = `<div class="kv"><span class="k">verdict</span><span class="mono">${esc(p.verdict || "?")}</span></div>`;
    if (p.operator_payload) h += `<div class="banner banner-info">${esc(p.operator_payload)}</div>`;
    const checks = (p.checks || []).map((c) => ({ check: c.name, status: c.status, detail: c.detail }));
    if (checks.length) h += `<div class="section-title">Checks</div>${table(checks)}`;
    if ((p.counterintuitive_findings || []).length) h += `<div class="banner banner-warn"><b>Counterintuitive findings</b> (contradict predicted direction)${ul(p.counterintuitive_findings)}</div>`;
    if ((p.anomalies || []).length) h += `<div class="banner banner-warn"><b>Anomalies</b>${ul(p.anomalies)}</div>`;
    return h;
  }

  function stageDraft(p) {
    const r = p.review || p;
    let h = `<div class="kv"><span class="k">recommendation</span><span class="mono">${esc(p.recommendation ?? r.recommendation ?? "?")}</span>
      <span class="k">score</span><span class="mono">${esc(p.score ?? r.score ?? "?")}</span></div>`;
    if (r.summary) h += `<div class="small">${esc(r.summary)}</div>`;
    if ((r.strengths || []).length) h += `<div class="section-title">Strengths</div>${ul(r.strengths)}`;
    if ((r.weaknesses || []).length) {
      h += `<div class="section-title">Weaknesses</div>` + (r.weaknesses).map((w) => {
        const sev = (w && w.severity) || "?", issue = (w && w.issue) || (typeof w === "string" ? w : "?"), fix = (w && w.suggested_fix) || "";
        return `<div class="small"><span class="sev sev-${esc(String(sev).toLowerCase())}">${esc(sev)}</span> ${esc(issue)}${fix ? ` <span class="muted">↳ ${esc(fix)}</span>` : ""}</div>`;
      }).join("");
    }
    if ((r.requested_changes || []).length) h += `<div class="section-title">Requested changes</div>${ul(r.requested_changes)}`;
    if ((r.missed_pitfalls || []).length) h += `<div class="banner banner-warn"><b>Missed pitfalls</b>${ul(r.missed_pitfalls)}</div>`;
    return h;
  }

  // ---- data refresh ----------------------------------------------------
  async function refreshOverview() {
    try {
      const ov = await api.get("/api/overview");
      state.runs = ov.runs || [];
      state.pending = ov.pending || [];
      renderBudget(ov.budget);
      renderPending();
      renderRuns();
      if (!state.selectedRunId && ov.active_run_id) await selectRun(ov.active_run_id);
    } catch (e) { /* keep last good */ }
  }

  // ---- SSE -------------------------------------------------------------
  function connectStream() {
    const conn = $("#conn");
    const es = new EventSource("/api/stream");
    es.addEventListener("ready", () => setConn("live", "live"));
    es.addEventListener("update", async (e) => {
      let d; try { d = JSON.parse(e.data); } catch { return; }
      if (d.budget != null) renderBudgetSpent(d.budget);
      if (d.runs) refreshOverview();
      if (d.checkpoints) { refreshOverview(); if (state.openCpId) refreshOpenCp(); if (state.detail) refreshDetail(); }
      if (d.messages && d.messages.runs && state.selectedRunId && d.messages.runs.includes(state.selectedRunId)) {
        await loadFeed(false);
        // keep the "now" line fresh without a full detail refetch
        if (!d.runs && !d.checkpoints) refreshNowLightweight();
      }
    });
    es.onerror = () => { setConn("down", "reconnecting…"); };
    es.onopen = () => setConn("live", "live");
    function setConn(stateName, label) {
      conn.dataset.state = stateName;
      conn.querySelector(".conn-label").textContent = label;
    }
  }

  function renderBudgetSpent(spent) {
    $("#budget-spent").textContent = fmtUsd2(spent);
    const cap = parseFloat(($("#budget-cap").textContent || "$0").slice(1)) || 0;
    if (cap) { const pct = Math.min(1, spent / cap); const fill = $("#budget-fill");
      fill.style.width = pct * 100 + "%";
      fill.classList.toggle("warn", pct >= 0.75 && pct < 0.92);
      fill.classList.toggle("danger", pct >= 0.92); }
  }

  async function refreshNowLightweight() {
    // Cheap: only the last event + current agent come from detail; refetch detail
    // is fine (small query) but throttle to avoid storms.
    if (refreshNowLightweight._t) return;
    refreshNowLightweight._t = setTimeout(async () => {
      refreshNowLightweight._t = null;
      await refreshDetail();
    }, 600);
  }

  async function refreshOpenCp() {
    if (!state.openCpId) return;
    // Don't clobber the operator mid-keystroke: if a field in the modal has
    // focus, skip this refresh (the SSE event will fire again, or the manual
    // submit will re-render).
    const ae = document.activeElement;
    if (ae && $("#modal-body") && $("#modal-body").contains(ae) &&
        (ae.tagName === "TEXTAREA" || ae.tagName === "INPUT")) return;
    try { const cp = await api.get(`/api/checkpoints/${state.openCpId}`); renderCheckpoint(cp); }
    catch (e) { /* ignore */ }
  }

  // ---- start a new run -------------------------------------------------
  async function openNewRunModal() {
    openModal("Start a new run", "", `<div class="skeleton" style="height:160px"></div>`);
    let data;
    try { data = await api.get("/api/projects"); }
    catch (e) { $("#modal-body").innerHTML = `<div class="banner banner-warn">Could not load projects.</div>`; return; }
    const projects = data.projects || [];
    const defAgent = data.default_agent || "lit_review";
    if (!projects.length) {
      $("#modal-body").innerHTML = `<div class="banner banner-warn">No launchable projects found under <span class="mono">projects/</span> (each needs a <span class="mono">config.toml</span>).</div>`;
      return;
    }
    const descById = Object.fromEntries(projects.map((p) => [p.id, p.description || ""]));
    $("#modal-body").innerHTML =
      `<div class="newrun-form">
         <div class="field">
           <label>Project</label>
           <select id="nr-project">${projects.map((p) =>
             `<option value="${esc(p.id)}">${esc(p.id)}${p.has_payload ? "" : "  ·  no kickoff payload"}</option>`).join("")}</select>
           <div class="muted small" id="nr-desc"></div>
         </div>
         <div class="field">
           <label>Starting agent</label>
           <input type="text" id="nr-agent" value="${esc(defAgent)}" />
         </div>
         <div class="field">
           <label>Kickoff payload <span class="muted">(prefilled from the project — edit if you like)</span></label>
           <textarea id="nr-payload" class="mono-area" style="min-height:200px" placeholder="(no kickoff_payload.json — paste a payload or leave blank)"></textarea>
         </div>
         <div class="form-actions">
           <button class="btn btn-primary" id="nr-start">Start run</button>
           <span class="muted small">Launches the runner; it pauses at checkpoint 1 (idea selection).</span>
         </div>
       </div>`;
    const sel = $("#nr-project");
    async function loadPayload(pid) {
      $("#nr-desc").textContent = descById[pid] || "";
      const t = $("#nr-payload"); t.value = "";
      try { const pr = await api.get(`/api/projects/${pid}/payload`); t.value = pr.payload; }
      catch (e) { /* no payload file — leave blank */ }
    }
    sel.onchange = () => loadPayload(sel.value);
    await loadPayload(sel.value);

    $("#nr-start").onclick = async () => {
      const btn = $("#nr-start"); btn.disabled = true;
      const before = state.runs[0] ? state.runs[0].run_id : null;
      const body = { project: sel.value, agent: $("#nr-agent").value.trim() || defAgent, payload: $("#nr-payload").value };
      const { ok, data } = await api.post("/api/runs/start", body);
      btn.disabled = false;
      if (ok && data.ok) {
        toast(`Started a run on ${sel.value}`, "success");
        closeModal();
        // The runner creates the run row immediately; pick it up + select it.
        setTimeout(async () => {
          await refreshOverview();
          const top = state.runs[0];
          if (top && top.run_id !== before) selectRun(top.run_id);
        }, 700);
      } else {
        toast((data && data.error) || "Could not start run", "error");
      }
    };
  }

  // ---- mobile nav ------------------------------------------------------
  function setupNav() {
    const btn = document.createElement("button");
    btn.className = "btn btn-ghost btn-sm sidebar-toggle";
    btn.textContent = "☰";
    btn.style.marginRight = "8px";
    btn.onclick = () => document.body.classList.toggle("nav-open");
    $(".topbar").insertBefore(btn, $(".brand"));
  }
  function closeNav() { document.body.classList.remove("nav-open"); }

  // ---- boot ------------------------------------------------------------
  // Snapshot mode (?snapshot): render the full initial state but skip the
  // live SSE stream and periodic timer. Lets a headless browser reach
  // network-idle for a static capture (and is handy for embedding a still).
  const SNAPSHOT = location.search.includes("snapshot");
  async function boot() {
    setupNav();
    syncTabs();
    $("#new-run-btn").onclick = openNewRunModal;
    ensureModels();   // warm the model-picker catalog (non-blocking)
    await refreshOverview();
    if (SNAPSHOT) {
      const params = new URLSearchParams(location.search);
      const wantRun = params.get("run");
      if (wantRun) await selectRun(wantRun);
      const wantTab = params.get("tab");
      if (wantTab) {
        state.tab = wantTab;
        syncTabs();
        if (wantTab === "handoffs") {
          await loadTimeline();
          // reveal the inbound prompt (first block) of each agent card
          $$("#panel-handoffs .act-card").forEach((card) => {
            const b = card.querySelector(".block");
            if (b) b.classList.add("open");
          });
        }
        if (wantTab === "checkpoints") await loadCheckpointsHistory();
        if (wantTab === "agents") renderAgents();
      }
      const c = $("#conn");
      c.dataset.state = "live";
      c.querySelector(".conn-label").textContent = "live";
      if (params.get("newrun")) await openNewRunModal();
      const wantCp = params.get("cp");
      if (wantCp) await openCheckpoint(wantCp);
      return;
    }
    connectStream();
    // periodic relative-time refresh + safety net if SSE stalls
    setInterval(() => { renderRuns(); }, 30000);
  }
  boot();
})();
