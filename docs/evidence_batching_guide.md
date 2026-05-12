# Evidence Batching Guide (25 Batches)

This repo session has `.git` mounted read-only, so we cannot `git add`/commit from inside this environment. The workaround is to apply the generated patch files from a normal checkout (where `.git` is writable), then commit in two commits.

## What Was Generated

Patch files (timestamp-free, stable paths):

- `reports/patches/0001_evidence_workplan_and_cli.patch`
- `reports/patches/0002_replay_batch_and_merge_helpers.patch`

These patches include:

1. `evidence-workplan` CLI command to produce 25 small work queues and evidence templates.
2. `replay-batch` CLI command to run seed -> merge -> stats -> compare -> resolver-on-replay -> dashboard in one command.
3. `merge_replay_files()` helper in `src/places_attr_conflation/harness.py` so we can merge explicit replay files without copying them into a directory.
4. CLI tests in `tests/test_harness_cli.py` for the new commands.

## Apply + Commit (Two Commits)

Run these commands in a normal clone of `MLAttributes` with a writable `.git`:

```bash
cd /path/to/MLAttributes

# apply patch 1 (workplan)
git apply reports/patches/0001_evidence_workplan_and_cli.patch
git add -A
git commit -m "Add evidence-workplan queues + templates"

# apply patch 2 (replay-batch + merge helper)
git apply reports/patches/0002_replay_batch_and_merge_helpers.patch
git add -A
git commit -m "Add replay-batch wrapper and merge_replay_files"

# verify
python3 -m unittest discover -s tests
```

If `git apply` fails due to drift, use:

```bash
git apply --3way reports/patches/0001_evidence_workplan_and_cli.patch
git apply --3way reports/patches/0002_replay_batch_and_merge_helpers.patch
```

## Workflow: Run 25 Evidence Batches

The conflict dork batches already exist under:

`reports/ranker/conflict_dorks_20260512_032836_536355_batches/`

### Step 1: Generate Workplan (25 Batches, 25 Case-Attributes Each)

```bash
python3 scripts/run_harness.py evidence-workplan \
  --batch-dir reports/ranker/conflict_dorks_20260512_032836_536355_batches \
  --batches 25 \
  --cases-per-batch 25
```

This writes a timestamped directory under:

`reports/replay_collected/evidence_workplan_<timestamp>/`

Each workplan batch includes:

- `batch_###.csv`: the retrieval/dork queue (10s to 100s of rows)
- `evidence_template_###.csv`: one row per `(case_id, attribute)` to fill with public evidence

### Step 2: Fill Evidence Template (Human)

For each `evidence_template_###.csv`, fill only public, authoritative sources:

- official business site pages (contact/location/services/about)
- government pages
- business registries
- (optionally) OSM for corroboration

Columns to fill (minimum):

- `url`
- `title` (optional but helpful)
- `page_text` or short snippet (paraphrased is fine)
- `source_type` (e.g. `official_site`, `government`, `business_registry`, `osm`, `unknown`)
- `extracted_value` (the value you want the resolver to choose for this attribute)

### Step 3: Import + Merge + Evaluate in One Command

For each completed template:

```bash
python3 scripts/run_harness.py replay-batch \
  --batch reports/replay_collected/evidence_workplan_<timestamp>/batch_001.csv \
  --evidence reports/replay_collected/evidence_workplan_<timestamp>/evidence_template_001.csv
```

This writes:

- new seed replay JSON under `reports/replay_collected/`
- new merged replay under `reports/replay/merged_<timestamp>.json`
- reports under:
  - `reports/replay_stats/replay_stats_<timestamp>.json`
  - `reports/retrieval_compare/compare_<timestamp>.json`
  - `reports/resolver_replay/resolver_on_replay_<timestamp>.json`
- refreshed dashboard under `reports/dashboard/`

### Step 4: Track Deltas Over Time

For each batch import, use:

- `compare_<timestamp>.json`:
  - `targeted.authoritative_found_rate` vs `fallback.authoritative_found_rate`
  - `deltas.authoritative_found_rate`
  - `targeted.citation_precision_proxy`
- `resolver_on_replay_<timestamp>.json`:
  - `accuracy`, `abstention_rate`, `high_confidence_wrong_rate`
  - `per_attribute` breakdown for `website`, `name`, `category`

Only claim retrieval/resolver improvement when these reports show it on non-trivial sample size.

## Next Token Window: Concrete Checklist

1. Fill `evidence_template_001.csv` with 10-25 high-quality cases (official sites first).
2. Run `replay-batch` and record the three report paths.
3. Repeat for `evidence_template_002.csv` ... `evidence_template_025.csv`.
4. After 5+ batches, re-check dashboard and compare deltas batch-to-batch.

