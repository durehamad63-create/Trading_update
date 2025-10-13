# Scan & Remove Synthetic Data — Runbook


This runbook explains how to detect synthetic data and remove it safely.


## 1. Automated scanning (recommended)
- Run the `data-sanity-check` job (see `ci/data-sanity-check.yml`).
- Checks performed:
- Duplicate burst patterns (identical rows repeated at scale).
- Uniform distributions where diversity is expected (e.g., identical timestamps, identical IDs in high cardinality fields).
- Known synthetic markers (metadata flags, placeholder strings like "lorem", "test@example", `synthetic_*`).
- Statistical anomaly detection (z-score, entropy metrics) — if a column entropy < threshold, flag for review.


## 2. Manual review
- If flagged, open a `data-investigation` issue with sample rows, dataset path, and scan output.
- Tag `@data-steward` and `@ml-owner`.


## 3. Removal steps
1. Quarantine dataset or snapshot (move to `quarantine/` with metadata).
2. If the dataset is in a pipeline, pause the pipeline and create a PR that removes the dataset reference.
3. Request replacement data from data owner; if none available, rollback to the last validated snapshot.


## 4. Re-validation
- When replacement arrives or synthetic data is removed, run full validation and publish results in the issue.