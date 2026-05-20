# Evidence URL Missing Report

Evidence fetching was not run.

## Inputs Checked
- `reports/workplans/pac_v1_first50/evidence_template_001.csv`
- `reports/workplans/pac_v1_first50/evidence_template_002.csv`

## URL Coverage
- Template rows checked: 50
- Rows with nonempty URL fields: 0
- Missing URL fields: 50
- Minimum required before fetch: 25

## Next Action
Fill at least 25 explicit evidence URLs in the two evidence templates, using `evidence_url_todo_001.md` and `evidence_url_todo_002.md` as the human review checklist. Then rerun `scripts/fetch_evidence_pages.py` with `--extend-schema`.
