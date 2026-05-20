# PAC v1 First50 Workplan Quality Notes

## Input
- Input file used: `data/conflict_dorks_20260512_032836_536355.csv`
- Candidate rows: 30,462
- Candidate case/attribute pairs: 5,077

## Selection
- Selected case/attribute count: 50
- Selected rows: 300
- Selected priority buckets: `P0_WEBSITE_BASELINE_WRONG`: 50 case/attribute pairs
- Selected attribute mix: `website`: 300 raw batch rows
- Evidence template rows: 50

## Audit
- Website case share: 1.00
- P0 case share: 1.00
- Missing priority buckets: 0
- Empty queries: 0
- Raw batch duplicate query rate: 0.98
- Evidence template duplicate query rate: 0.24
- Max same template query occurrences: 3
- Template duplicate cap respected: yes

## Decision
The first50 workplan is acceptable for human URL collection. The raw batch rows still contain repeated candidate queries, but the pilot gate now evaluates the evidence template rows used for collection. Template query selection respects the configured duplicate cap of 3.

## Exact Next Action
Use `evidence_url_todo_001.md` and `evidence_url_todo_002.md` to fill explicit URLs in `evidence_template_001.csv` and `evidence_template_002.csv`. Do not fetch until at least 25 evidence template rows have nonempty `url` fields.
