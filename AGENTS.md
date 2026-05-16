# AGENTS.md — MLAttributes / ProjectTerra PAC

## Project identity

MLAttributes is a retrieval-aware Place Attribute Conflation system. It is not generic record linkage. It resolves conflicting POI attributes using replayable evidence, source authority, freshness/staleness scoring, identity drift labels, resolver abstention, and benchmark reports.

## Non-negotiable constraints

Do not create a parallel architecture. Extend the existing replay/harness/dork/collector/evidence pipeline first.

Prefer extending:

- `src/places_attr_conflation/replay.py`
- `src/places_attr_conflation/harness.py`
- `scripts/run_harness.py`
- `src/places_attr_conflation/conflict_dorks.py`
- `src/places_attr_conflation/dorking.py`
- `src/places_attr_conflation/collector.py`
- `src/places_attr_conflation/collector_static.py`
- `scripts/fetch_evidence_pages.py`
- `src/places_attr_conflation/overture_context.py`
- `src/places_attr_conflation/golden.py`

Only add new files when the responsibility is genuinely new, such as a review dashboard, website evidence enrichment helper, identity classifier, or corpus builder.

## Corpus v1 target

Build toward a 650-case replay corpus:

- 425 website-heavy cases
- 100 identity-drift cases
- 60 category/taxonomy cases
- 40 source-dependency / aggregator-echo cases
- 25 abstention / weak-evidence / tie cases

Website is primary. Category and name are secondary. Phone and address are supporting signals.

## Confirmed website labels

- `OFFICIAL_CURRENT`
- `OFFICIAL_STALE`
- `OFFICIAL_DEAD`
- `OFFICIAL_WRONG_ENTITY`
- `OFFICIAL_CHAIN_ONLY`
- `OFFICIAL_LOCATION_PAGE`
- `SOCIAL_ONLY_CURRENT`
- `AGGREGATOR_ONLY`
- `PARKED_DOMAIN`
- `NO_WEBSITE_FOUND`
- `AMBIGUOUS_WEBSITE`

## Confirmed identity labels

- `SAME_ENTITY`
- `MOVED_ENTITY`
- `RENAMED_ENTITY`
- `OWNERSHIP_CHANGE`
- `NEW_ENTITY_SAME_ADDRESS`
- `STALE_OFFICIAL_SITE`
- `BRANCH_AMBIGUITY`
- `TEMPORARY_CLOSURE`
- `PERMANENT_CLOSURE`
- `UNKNOWN_IDENTITY`

## Required behavior

All corpus behavior must be deterministic, replayable, backwards compatible with old replay files, visible in `replay-stats`, evaluable through website-authority and PAC benchmark commands, and release-gated.

## Testing

Run targeted tests after each phase. If tests do not exist, add small deterministic tests before changing behavior. Do not rely on live network tests for CI.
