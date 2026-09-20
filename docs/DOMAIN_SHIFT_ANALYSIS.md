# Domain Shift Analysis: Synthetic vs. Indian ED Data

## Status: pending Indian dataset access

This document describes the **methodology** CareConnect uses to compare two
datasets in the common schema (`ml.experiments.domain_shift.compare_distributions`),
and states plainly that the actual synthetic-vs-Indian comparison has **not
been run**, because no Indian patient-level dataset is available to this
repository (see `RESEARCH_DOCUMENTATION.md` for what was checked). Every
number below that would require Indian data is marked `TBD`.

## What gets compared

`compare_distributions(df_a, df_b)` produces, for any two datasets in the
common schema:

1. **Numeric distributions** - `age`, `heart_rate`, `respiratory_rate`,
   `systolic_bp`, `diastolic_bp`, `oxygen_saturation`, `temperature`,
   `pain_level`: mean, std, quartiles, and the mean difference between the
   two datasets.
2. **Categorical distributions** - `sex`, `arrival_transport`: normalized
   value frequencies side by side.
3. **Text vocabulary** - `chief_complaint`: top-N word frequencies for each
   dataset and the fraction of top terms that overlap. This is a
   dependency-free word-frequency check, not a real TF-IDF comparison (that
   only exists once a vectorizer is fit on a specific training split).
4. **Target distribution** - ESI 1-5 class counts/percentages for each
   dataset (`ml.experiments.reports.class_distribution_report`).

This function is implemented and unit-tested
(`ml/tests/test_experiments.py::test_compare_distributions_runs_and_reports_target_distributions`)
against two synthetic samples purely to confirm the mechanism runs and
returns the expected structure - **that is not a domain-shift finding**, just
a check that the code works.

## Synthetic dataset's own target distribution (n=12000, seed=42)

From `ml/experiments/reports/class_distribution.csv`, the one half of this
comparison that can actually be reported today:

| ESI | Count | Percent |
|---|---:|---:|
| 1 | 900 | 7.50% |
| 2 | 2123 | 17.69% |
| 3 | 4792 | 39.93% |
| 4 | 1781 | 14.84% |
| 5 | 2404 | 20.03% |

This reflects the hand-built generative process in `ml/data/synthetic.py`
(`_SEVERITY_P`), not a real ED's triage distribution - it exists to be
usable for pipeline development, not to be representative.

## Indian dataset's distribution

**TBD.** No comparison numbers exist yet. When a real Indian dataset is
obtained and loaded via `ml.data.indian.load_indian_dataset`, run:

```bash
python -m ml.experiments.run_domain_experiment --experiment all \
    --source indian --data-path <path to the real file>
```

This populates `ml/experiments/reports/data_quality.json`'s `"indian"` key
(currently the literal string `"NOT_AVAILABLE"`) and enables a real call to
`compare_distributions(synthetic_df, indian_df)`.

## What to watch for once real data exists (not yet observed)

Documented here as *expectations to test*, not findings:

- **Triage scale**: several Indian ED protocols in the literature (e.g. the
  AIIMS Triage Protocol) are not the 5-level ESI scale. If the obtained
  dataset uses a different scale, `ml.data.indian.load_indian_dataset`
  requires an explicit `esi_mapping` - it will not silently coerce one scale
  into another.
- **Vitals recording conventions**: temperature units, blood pressure
  notation, and which vitals are routinely recorded at triage can differ
  from a US ED (MIMIC-IV-ED) or the synthetic generator's assumptions.
- **Chief-complaint language**: code-mixed language (e.g. Hindi/English or
  regional-language terms transliterated into English) is very likely to
  reduce the TF-IDF vocabulary overlap with the synthetic dataset's English
  medical-English phrasing, which could weaken exactly the text signal that
  Experiment 10 (structured-vs-text ablation) found helpful on synthetic
  data.
- **Class balance**: EMcounter's published description does not report an
  ESI-equivalent distribution, so there is no prior expectation to compare
  against yet.

## Interpreting a future result

When this comparison is eventually run, read differences as *where the two
distributions differ*, not as a ranking of which dataset is "more correct" -
the synthetic dataset is a development aid, not a ground truth to defend.
