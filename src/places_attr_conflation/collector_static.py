"""Static replay collector generator (no local server required).

Some environments restrict binding to localhost ports. This generator writes a
single self-contained HTML file with embedded batch payload. The page lets a
human paste authoritative evidence and then downloads a replay corpus JSON file.
"""

from __future__ import annotations

import json
from pathlib import Path

from .collector import load_collector_cases, collector_cases_payload


_TEMPLATE = """<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width,initial-scale=1" />
    <title>Replay Collector (Static)</title>
    <style>
      :root { color-scheme: light; }
      body { font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; margin: 16px; line-height: 1.35; }
      h1 { font-size: 18px; margin: 0 0 12px; }
      .row { display: flex; gap: 10px; flex-wrap: wrap; align-items: center; }
      .pill { display: inline-block; padding: 2px 8px; border: 1px solid #ddd; border-radius: 8px; font-size: 12px; background: #fafafa; }
      .grid { display: grid; grid-template-columns: 360px 1fr; gap: 12px; align-items: start; }
      @media (max-width: 980px) { .grid { grid-template-columns: 1fr; } }
      .panel { border: 1px solid #e5e5e5; border-radius: 8px; padding: 10px; background: #fff; }
      .list { max-height: 70vh; overflow: auto; }
      .case { padding: 8px; border-radius: 8px; cursor: pointer; }
      .case:hover { background: #f4f8ff; }
      .case.active { background: #eaf2ff; outline: 1px solid #cfe3ff; }
      .small { font-size: 12px; color: #444; }
      .muted { color: #666; }
      table { width: 100%; border-collapse: collapse; }
      th, td { border-bottom: 1px solid #eee; padding: 6px; vertical-align: top; font-size: 13px; }
      th { text-align: left; background: #fafafa; position: sticky; top: 0; }
      input, textarea, select { width: 100%; box-sizing: border-box; padding: 6px; border-radius: 6px; border: 1px solid #ddd; font: inherit; }
      textarea { min-height: 56px; }
      button { border: 1px solid #ddd; background: #fff; border-radius: 8px; padding: 6px 10px; cursor: pointer; }
      button.primary { border-color: #2b6cb0; background: #2b6cb0; color: #fff; }
      button:disabled { opacity: 0.5; cursor: not-allowed; }
      .k { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 12px; }
      .warn { color: #9a3412; }
    </style>
  </head>
  <body>
    <h1>Replay Collector (Static)</h1>
    <div class="row small muted" id="meta"></div>
    <div class="grid">
      <div class="panel list">
        <div class="row" style="margin-bottom:8px;">
          <input id="filter" placeholder="Filter by attribute or id..." />
          <span class="pill" id="counts">0 cases</span>
        </div>
        <div id="caseList"></div>
      </div>
      <div class="panel">
        <div id="caseDetail" class="muted">Select a case.</div>
      </div>
    </div>

    <script>
      const BATCH = __CASES_JSON__;
      const state = { cases: BATCH.cases || [], activeIdx: -1, edits: {} };
      function esc(s){ return String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])); }
      function googleLink(q){ return "https://www.google.com/search?q=" + encodeURIComponent(q); }

      function renderList(){
        const f = document.getElementById("filter").value.trim().toLowerCase();
        const list = document.getElementById("caseList");
        list.innerHTML = "";
        let shown = 0;
        state.cases.forEach((c, idx) => {
          const hay = (c.case_id + " " + c.attribute).toLowerCase();
          if (f && !hay.includes(f)) return;
          shown++;
          const div = document.createElement("div");
          div.className = "case" + (idx === state.activeIdx ? " active" : "");
          div.onclick = () => { state.activeIdx = idx; renderList(); renderDetail(); };
          div.innerHTML = `
            <div class="row">
              <span class="pill">${esc(c.attribute)}</span>
              <span class="k">${esc(c.case_id.slice(0,12))}</span>
            </div>
            <div class="small muted">truth: ${esc(c.truth || "")}</div>
            <div class="small muted">pred: ${esc(c.prediction || "")}</div>
          `;
          list.appendChild(div);
        });
        document.getElementById("counts").textContent = shown + " cases";
      }

      function ensureEdits(caseId, queries){
        if (state.edits[caseId]) return state.edits[caseId];
        state.edits[caseId] = { attempts: (queries||[]).map(q => ({ layer: q.layer || "fallback", query: q.query || "", fetched_pages: [] })) };
        return state.edits[caseId];
      }

      function addPage(caseId, attemptIdx){
        const edit = state.edits[caseId];
        edit.attempts[attemptIdx].fetched_pages.push({
          url: "", title: "", page_text: "", source_type: "unknown", extracted_values: {},
          recency_days: null, zombie_score: 0.0, identity_change_score: 0.0, notes: ""
        });
        renderDetail();
      }

      function setPageField(caseId, attemptIdx, pageIdx, field, value){
        const page = state.edits[caseId].attempts[attemptIdx].fetched_pages[pageIdx];
        if (field === "recency_days") page.recency_days = value === "" ? null : Number(value);
        else if (field === "zombie_score" || field === "identity_change_score") page[field] = value === "" ? 0.0 : Number(value);
        else page[field] = value;
      }

      function setExtracted(caseId, attemptIdx, pageIdx, attr, value){
        const page = state.edits[caseId].attempts[attemptIdx].fetched_pages[pageIdx];
        page.extracted_values = page.extracted_values || {};
        if (!value) delete page.extracted_values[attr];
        else page.extracted_values[attr] = value;
      }

      function renderDetail(){
        const root = document.getElementById("caseDetail");
        if (state.activeIdx < 0) { root.textContent = "Select a case."; return; }
        const c = state.cases[state.activeIdx];
        const edit = ensureEdits(c.case_id, c.queries || []);
        root.innerHTML = `
          <div class="row" style="justify-content:space-between; align-items:flex-start;">
            <div>
              <div class="row"><span class="pill">${esc(c.attribute)}</span><span class="k">${esc(c.case_id)}</span></div>
              <div class="small muted">truth: <span class="k">${esc(c.truth || "")}</span></div>
              <div class="small muted">prediction: <span class="k">${esc(c.prediction || "")}</span> <span class="pill">${esc(c.baseline || "")}</span></div>
              <div class="small muted">preferred sources: ${esc(c.preferred_sources || "")}</div>
            </div>
            <div class="row">
              <button class="primary" id="downloadBtn">Download Replay JSON</button>
            </div>
          </div>
          <div style="margin-top:10px;">
            <table>
              <thead><tr><th style="width:160px;">Query</th><th>Evidence Pages</th></tr></thead>
              <tbody id="attemptsBody"></tbody>
            </table>
          </div>
          <div id="dlStatus" class="small muted" style="margin-top:10px;"></div>
        `;

        const tbody = root.querySelector("#attemptsBody");
        edit.attempts.forEach((a, ai) => {
          const tr = document.createElement("tr");
          const qCell = document.createElement("td");
          qCell.innerHTML = `
            <div class="pill">${esc(a.layer)}</div>
            <div class="small k" style="margin-top:6px; word-break:break-word;">${esc(a.query)}</div>
            <div style="margin-top:6px;"><a class="small" href="${googleLink(a.query)}" target="_blank" rel="noreferrer">open search</a></div>
            <div style="margin-top:8px;"><button type="button" data-add="${ai}">+ Add Page</button></div>
          `;
          const eCell = document.createElement("td");
          eCell.innerHTML = a.fetched_pages.length ? "" : `<span class="small muted">No pages yet.</span>`;
          a.fetched_pages.forEach((p, pi) => {
            const div = document.createElement("div");
            div.className = "panel";
            div.style.marginBottom = "8px";
            div.innerHTML = `
              <div class="row">
                <div style="flex:2;">
                  <div class="small muted">url</div>
                  <input value="${esc(p.url || "")}" data-f="url" data-ai="${ai}" data-pi="${pi}" placeholder="https://..." />
                </div>
                <div style="flex:1;">
                  <div class="small muted">source_type</div>
                  <select data-f="source_type" data-ai="${ai}" data-pi="${pi}">
                    ${["official_site","government","business_registry","trusted_directory","social","aggregator","unknown"].map(x => `<option value="${x}" ${p.source_type===x?"selected":""}>${x}</option>`).join("")}
                  </select>
                </div>
              </div>
              <div class="row" style="margin-top:8px;">
                <div style="flex:1;">
                  <div class="small muted">title</div>
                  <input value="${esc(p.title || "")}" data-f="title" data-ai="${ai}" data-pi="${pi}" />
                </div>
              </div>
              <div class="row" style="margin-top:8px;">
                <div style="flex:1;">
                  <div class="small muted">snippet/page_text (paste the line with the attribute)</div>
                  <textarea data-f="page_text" data-ai="${ai}" data-pi="${pi}">${esc(p.page_text || "")}</textarea>
                </div>
              </div>
              <div class="row" style="margin-top:8px;">
                <div style="flex:1;">
                  <div class="small muted">extracted_${esc(c.attribute)} (normalized)</div>
                  <input value="${esc((p.extracted_values||{})[c.attribute] || "")}" data-ex="${esc(c.attribute)}" data-ai="${ai}" data-pi="${pi}" />
                </div>
                <div style="flex:1;">
                  <div class="small muted">recency_days (optional)</div>
                  <input value="${p.recency_days==null ? "" : esc(p.recency_days)}" data-f="recency_days" data-ai="${ai}" data-pi="${pi}" />
                </div>
              </div>
            `;
            eCell.appendChild(div);
          });
          tr.appendChild(qCell);
          tr.appendChild(eCell);
          tbody.appendChild(tr);
        });

        root.querySelectorAll("button[data-add]").forEach(btn => { btn.onclick = () => addPage(c.case_id, Number(btn.getAttribute("data-add"))); });
        root.querySelectorAll("input[data-f], textarea[data-f], select[data-f]").forEach(el => {
          el.oninput = () => setPageField(c.case_id, Number(el.getAttribute("data-ai")), Number(el.getAttribute("data-pi")), el.getAttribute("data-f"), el.value);
          el.onchange = el.oninput;
        });
        root.querySelectorAll("input[data-ex]").forEach(el => {
          el.oninput = () => setExtracted(c.case_id, Number(el.getAttribute("data-ai")), Number(el.getAttribute("data-pi")), el.getAttribute("data-ex"), el.value);
        });

        root.querySelector("#downloadBtn").onclick = () => downloadReplay(c);
      }

      function buildReplayCorpus(){
        const episodes = [];
        for (const c of state.cases){
          const edit = ensureEdits(c.case_id, c.queries || []);
          episodes.push({ case_id: c.case_id, attribute: c.attribute, place: { base_id: c.base_id }, gold_value: c.truth || "", search_attempts: edit.attempts });
        }
        return { schema_version: 1, episodes };
      }

      function downloadReplay(activeCase){
        const replay = buildReplayCorpus();
        const fileName = "replay_collected_" + (activeCase.attribute || "attr") + "_" + (new Date()).toISOString().replace(/[:.]/g,"") + ".json";
        const blob = new Blob([JSON.stringify(replay, null, 2)], {type:"application/json"});
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = fileName;
        a.click();
        URL.revokeObjectURL(url);
        document.getElementById("dlStatus").innerHTML = "Downloaded: <span class='k'>" + esc(fileName) + "</span>";
      }

      document.getElementById("meta").textContent = "Loaded " + state.cases.length + " conflict cases from this batch.";
      document.getElementById("filter").addEventListener("input", renderList);
      renderList();
    </script>
  </body>
</html>
"""


def write_static_collector_html(batch_csv: str | Path, out_html: str | Path) -> Path:
    cases = load_collector_cases(batch_csv)
    payload = collector_cases_payload(cases)
    html = _TEMPLATE.replace("__CASES_JSON__", json.dumps(payload))
    out = Path(out_html)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    return out

