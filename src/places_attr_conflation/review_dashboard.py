"""Static keyboard-driven replay review dashboard."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Iterable

from .corpus_labels import IDENTITY_LABELS, WEBSITE_LABELS
from .replay import ReplayEpisode, load_replay_corpus


SHORTCUTS = {
    "ArrowLeft": "mark rejected / no",
    "ArrowRight": "mark accepted / yes",
    "a": "expected_abstain = true",
    "z": "expected_abstain = false",
    "s": "identity_label = SAME_ENTITY",
    "m": "identity_label = MOVED_ENTITY",
    "r": "identity_label = RENAMED_ENTITY",
    "n": "identity_label = NEW_ENTITY_SAME_ADDRESS",
    "b": "identity_label = BRANCH_AMBIGUITY",
    "o": "website_label = OFFICIAL_CURRENT",
    "t": "website_label = OFFICIAL_STALE",
    "d": "website_label = OFFICIAL_DEAD",
    "g": "website_label = AGGREGATOR_ONLY",
    "p": "website_label = PARKED_DOMAIN",
    "c": "website_label = OFFICIAL_CHAIN_ONLY",
    "l": "website_label = OFFICIAL_LOCATION_PAGE",
    "Space": "skip",
    "Enter": "save and next",
}


HTML_TEMPLATE = r'''<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>__TITLE__</title>
  <style>
    :root { color-scheme: light dark; font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
    body { margin: 0; padding: 24px; background: Canvas; color: CanvasText; }
    .shell { max-width: 1200px; margin: 0 auto; }
    .topbar { display: flex; justify-content: space-between; gap: 16px; align-items: center; margin-bottom: 16px; }
    .card { border: 1px solid color-mix(in srgb, CanvasText 18%, transparent); border-radius: 16px; padding: 16px; margin: 12px 0; box-shadow: 0 1px 8px color-mix(in srgb, CanvasText 10%, transparent); }
    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(230px, 1fr)); gap: 12px; }
    .field { border: 1px solid color-mix(in srgb, CanvasText 14%, transparent); border-radius: 12px; padding: 10px; }
    .label { font-size: 12px; opacity: 0.7; text-transform: uppercase; letter-spacing: 0.04em; }
    .value { margin-top: 4px; word-break: break-word; }
    .page { border-left: 4px solid color-mix(in srgb, CanvasText 30%, transparent); padding: 10px; background: color-mix(in srgb, CanvasText 5%, transparent); border-radius: 8px; margin: 8px 0; }
    .muted { opacity: 0.72; }
    .toolbar { display: flex; flex-wrap: wrap; gap: 8px; margin: 12px 0; }
    button, select, input, textarea { font: inherit; }
    button { border-radius: 10px; border: 1px solid color-mix(in srgb, CanvasText 20%, transparent); padding: 8px 10px; cursor: pointer; }
    textarea { width: 100%; min-height: 72px; border-radius: 10px; padding: 8px; box-sizing: border-box; }
    .shortcuts { columns: 2; font-size: 13px; }
    .kbd { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; padding: 1px 5px; border: 1px solid color-mix(in srgb, CanvasText 24%, transparent); border-radius: 5px; }
  </style>
</head>
<body>
<div class="shell">
  <div class="topbar">
    <div><h1>__TITLE__</h1><div id="progress" class="muted"></div></div>
    <div class="toolbar">
      <button onclick="prevCase()">← Previous</button>
      <button onclick="nextCase()">Next →</button>
      <button onclick="downloadReviewed()">Download reviewed replay JSON</button>
    </div>
  </div>
  <div class="card"><strong>Keyboard shortcuts</strong><div id="shortcuts" class="shortcuts"></div></div>
  <div id="case"></div>
</div>
<script id="replay-data" type="application/json">__EPISODES__</script>
<script id="label-data" type="application/json">__LABELS__</script>
<script>
const episodes = JSON.parse(document.getElementById('replay-data').textContent);
const labelData = JSON.parse(document.getElementById('label-data').textContent);
let idx = 0;
function escapeHtml(value) { return String(value ?? '').replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c])); }
function pagesFor(ep) { return (ep.search_attempts || []).flatMap(a => (a.fetched_pages || []).map(p => Object.assign({}, p, {layer: a.layer, query: a.query}))); }
function renderShortcuts() { document.getElementById('shortcuts').innerHTML = Object.entries(labelData.shortcuts).map(([k,v]) => '<div><span class="kbd">' + escapeHtml(k) + '</span> ' + escapeHtml(v) + '</div>').join(''); }
function field(label, value) { return '<div class="field"><div class="label">' + escapeHtml(label) + '</div><div class="value">' + escapeHtml(value || '') + '</div></div>'; }
function selectField(label, key, values) {
  const ep = episodes[idx];
  const opts = [''].concat(values).map(v => '<option value="' + escapeHtml(v) + '" ' + (ep[key]===v?'selected':'') + '>' + escapeHtml(v) + '</option>').join('');
  return '<div class="field"><div class="label">' + escapeHtml(label) + '</div><select onchange="setField(\'' + key + '\', this.value)">' + opts + '</select></div>';
}
function renderCase() {
  if (!episodes.length) { document.getElementById('case').innerHTML = '<div class="card">No episodes loaded.</div>'; return; }
  const ep = episodes[idx];
  const pages = pagesFor(ep);
  document.getElementById('progress').textContent = String(idx + 1) + ' / ' + String(episodes.length) + ' cases';
  document.getElementById('case').innerHTML = '<div class="card"><h2>' + escapeHtml(ep.case_id) + ' <span class="muted">' + escapeHtml(ep.attribute) + '</span></h2><div class="grid">'
    + field('Gold value', ep.gold_value) + field('Expected decision', ep.expected_decision) + field('Expected abstain', ep.expected_abstain)
    + field('Review status', ep.review_status || 'unreviewed') + field('Current value', ep.place?.current_value) + field('Base value', ep.place?.base_value)
    + field('Name', ep.place?.name) + field('Address', ep.place?.address) + field('City / region', [ep.place?.city, ep.place?.region].filter(Boolean).join(', '))
    + field('Overture id', ep.place?.overture_id || ep.place?.gers_id) + '</div></div><div class="card"><h3>Labels</h3><div class="grid">'
    + field('Case type', ep.case_type) + selectField('Website label', 'website_label', labelData.website_labels)
    + selectField('Identity label', 'identity_label', labelData.identity_labels) + field('Truth source type', ep.truth_source_type)
    + field('Label origin', ep.label_origin) + field('Difficulty', ep.difficulty) + '</div><p><label><input type="checkbox" ' + (ep.expected_abstain === true ? 'checked' : '')
    + ' onchange="setField(\'expected_abstain\', this.checked)" /> Expected abstain</label></p><textarea placeholder="Reviewer notes" oninput="setField(\'reviewer_notes\', this.value)">'
    + escapeHtml(ep.reviewer_notes || '') + '</textarea></div><div class="card"><h3>Evidence pages (' + pages.length + ')</h3>'
    + pages.map(p => '<div class="page"><div><a href="' + escapeHtml(p.url) + '" target="_blank" rel="noreferrer">' + escapeHtml(p.title || p.url) + '</a></div><div class="muted">' + escapeHtml(p.source_type) + ' · ' + escapeHtml(p.evidence_role) + ' · ' + escapeHtml(p.source_family_id) + ' · ' + escapeHtml(p.layer) + '</div><div class="muted">Query: ' + escapeHtml(p.query) + '</div><p>' + escapeHtml((p.page_text || '').slice(0, 900)) + '</p></div>').join('')
    + '</div>';
}
function setField(key, value) { episodes[idx][key] = value; }
function mark(status) { episodes[idx].review_status = status; renderCase(); }
function nextCase() { idx = Math.min(episodes.length - 1, idx + 1); renderCase(); }
function prevCase() { idx = Math.max(0, idx - 1); renderCase(); }
function downloadReviewed() { const payload = {schema_version: 1, generated_at: new Date().toISOString(), episodes}; const blob = new Blob([JSON.stringify(payload, null, 2)], {type: 'application/json'}); const a = document.createElement('a'); a.href = URL.createObjectURL(blob); a.download = 'reviewed_replay.json'; a.click(); }
document.addEventListener('keydown', ev => {
  if (['INPUT','TEXTAREA','SELECT'].includes(document.activeElement.tagName)) return;
  const ep = episodes[idx];
  if (ev.key === 'ArrowLeft') { mark('rejected'); ev.preventDefault(); return; }
  if (ev.key === 'ArrowRight') { mark('accepted'); ev.preventDefault(); return; }
  if (ev.key === 'Enter') { nextCase(); ev.preventDefault(); return; }
  if (ev.key === ' ') { mark('skipped'); nextCase(); ev.preventDefault(); return; }
  const k = ev.key.toLowerCase();
  if (k === 'a') ep.expected_abstain = true;
  if (k === 'z') ep.expected_abstain = false;
  if (k === 's') ep.identity_label = 'SAME_ENTITY';
  if (k === 'm') ep.identity_label = 'MOVED_ENTITY';
  if (k === 'r') ep.identity_label = 'RENAMED_ENTITY';
  if (k === 'n') ep.identity_label = 'NEW_ENTITY_SAME_ADDRESS';
  if (k === 'b') ep.identity_label = 'BRANCH_AMBIGUITY';
  if (k === 'o') ep.website_label = 'OFFICIAL_CURRENT';
  if (k === 't') ep.website_label = 'OFFICIAL_STALE';
  if (k === 'd') ep.website_label = 'OFFICIAL_DEAD';
  if (k === 'g') ep.website_label = 'AGGREGATOR_ONLY';
  if (k === 'p') ep.website_label = 'PARKED_DOMAIN';
  if (k === 'c') ep.website_label = 'OFFICIAL_CHAIN_ONLY';
  if (k === 'l') ep.website_label = 'OFFICIAL_LOCATION_PAGE';
  renderCase();
});
renderShortcuts();
renderCase();
</script>
</body>
</html>'''


def _json_for_html(payload: object) -> str:
    return json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")


def build_review_dashboard_html(episodes: Iterable[ReplayEpisode], *, title: str = "Replay Review Dashboard") -> str:
    labels_payload = {
        "website_labels": sorted(WEBSITE_LABELS),
        "identity_labels": sorted(IDENTITY_LABELS),
        "shortcuts": SHORTCUTS,
    }
    return (
        HTML_TEMPLATE.replace("__TITLE__", html.escape(title))
        .replace("__EPISODES__", _json_for_html([episode.to_dict() for episode in episodes]))
        .replace("__LABELS__", _json_for_html(labels_payload))
    )


def write_review_dashboard(input_path: str | Path, output_dir: str | Path, *, title: str = "Replay Review Dashboard") -> dict[str, object]:
    episodes = load_replay_corpus(input_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    index_path = out_dir / "index.html"
    index_path.write_text(build_review_dashboard_html(episodes, title=title), encoding="utf-8")
    return {"input": str(input_path), "output": str(index_path), "episodes": len(episodes)}
