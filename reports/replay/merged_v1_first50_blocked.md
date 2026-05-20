# merged_v1_first50 Replay Assembly Blocked

`reports/replay/merged_v1_first50.json` was not created.

## Reason
The first50 evidence templates contain 50 rows and 0 explicit evidence URLs. Building a replay corpus now would preserve search attempts, but it would not attach evidence pages and would not prove enriched evidence can be replayed before human truth review.

## Existing Replay Commands Inspected
- `python scripts/run_harness.py replay-seed`
- `python scripts/run_harness.py replay-batch`
- `python scripts/run_harness.py replay-merge`
- `python scripts/run_harness.py resolver-on-replay`

The existing seed path is schema-valid, but it does not preserve workplan guess metadata into replay episode fields. `scripts/build_replay_from_workplan.py` was added for that compatibility path and only attaches pages when evidence rows contain explicit URLs.

## Exact Next Action
Fill at least 25 explicit URLs in `reports/workplans/pac_v1_first50/evidence_template_001.csv` and `reports/workplans/pac_v1_first50/evidence_template_002.csv`, run evidence fetching and auditing, then run:

```bash
python scripts/build_replay_from_workplan.py \
  --batch reports/workplans/pac_v1_first50/batch_001.csv \
  --evidence reports/workplans/pac_v1_first50/evidence_template_001.csv \
  --batch reports/workplans/pac_v1_first50/batch_002.csv \
  --evidence reports/workplans/pac_v1_first50/evidence_template_002.csv \
  --output reports/replay/merged_v1_first50.json
```

Then audit it with:

```bash
python scripts/audit_replay.py \
  --input reports/replay/merged_v1_first50.json \
  --output reports/replay/merged_v1_first50_audit.json
```
