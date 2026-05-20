# Live Search URL Evidence Finder

`scripts/find_evidence_urls.py` discovers candidate evidence URLs for PAC replay construction. Live search is used only to find candidate URLs. It does not decide truth labels, set `gold_value`, set expected decisions, or mark review accepted.

## Replay Artifacts

Every run writes replayable discovery artifacts under `--output-dir`:

- `search_queries.jsonl`: every generated query with case id, layer, provider, and timestamp.
- `search_results_snapshot.jsonl`: every returned result with URL, title, snippet, rank, provider, query, case id, and timestamp.
- `url_candidates.csv`: scored candidate URLs with source type, role guess, matching signals, rejection reason, and score.
- `url_finder_report.json`: run summary.
- `url_finder_notes.md`: human-readable notes.

The tool does not fetch candidate pages. Use `scripts/fetch_evidence_pages.py --extend-schema` after explicit URLs are selected or autofilled.

## Providers

Query-only mode records query snapshots without live search:

```bash
python scripts/find_evidence_urls.py \
  --input reports/workplans/pac_v1_first50/evidence_template_001.csv \
  --output-dir reports/workplans/pac_v1_first50/url_finder \
  --region ca-santa-cruz \
  --provider query-only
```

Brave Search uses `BRAVE_SEARCH_API_KEY`:

```bash
BRAVE_SEARCH_API_KEY=... python scripts/find_evidence_urls.py \
  --input reports/workplans/pac_v1_first50/evidence_template_001.csv \
  --output-dir reports/workplans/pac_v1_first50/url_finder \
  --region ca-santa-cruz \
  --provider brave \
  --limit 5
```

Google Programmable Search uses `GOOGLE_CSE_API_KEY` and `GOOGLE_CSE_ID`:

```bash
GOOGLE_CSE_API_KEY=... GOOGLE_CSE_ID=... python scripts/find_evidence_urls.py \
  --input reports/workplans/pac_v1_first50/evidence_template_001.csv \
  --output-dir reports/workplans/pac_v1_first50/url_finder \
  --region ca-santa-cruz \
  --provider google-cse \
  --limit 5
```

A local command provider accepts one JSONL request on stdin and returns JSONL results on stdout:

```bash
python scripts/find_evidence_urls.py \
  --input reports/workplans/pac_v1_first50/evidence_template_001.csv \
  --output-dir reports/workplans/pac_v1_first50/url_finder \
  --region ca-santa-cruz \
  --provider command \
  --search-command "python scripts/my_search_provider.py"
```

If live credentials are missing, the CLI falls back to query-only mode unless `--require-live-search` is passed.

## Autofill

Autofill never modifies the original template. It writes `*.autofilled.csv` files only when `--write-autofill` is passed.

A URL is autofilled only when it is high confidence, official/government/business-registry sourced, not rejected, not an unsupported generic chain homepage, and unambiguous for that case. Notes record `query_used`, `search_provider`, `search_rank`, and `evidence_role`.

## Guardrails

Do not treat search rank or candidate score as truth. Candidate rows are review aids only. Human review must finalize truth labels later.
