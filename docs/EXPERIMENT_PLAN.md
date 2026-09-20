# Experiment Plan

How to run every experiment in the domain-transfer research phase, what each
one is for, and its actual status as of this writing. Implementation:
`ml/experiments/run_domain_experiment.py`. Results actually produced by a run
are committed under `ml/experiments/results/*.json` and
`ml/experiments/reports/*.csv` - the numbers in this document are read from
those files, not typed in separately.

Four of the five experiment configs need "a real dataset" - the runner
takes that generically via `--source {mimic,indian}` + `--data-path`, so the
exact same code path answers "does this transfer to MIMIC-IV-ED?" today and
"does this transfer to an Indian dataset?" the moment one exists, without
duplicated logic.

```bash
python -m ml.experiments.run_domain_experiment --experiment all
python -m ml.experiments.run_domain_experiment --experiment synthetic_baseline
python -m ml.experiments.run_domain_experiment --experiment all --source mimic --data-path ml/data/mimic-iv-ed
python -m ml.experiments.run_domain_experiment --experiment all --source indian --data-path ml/data/indian/export.csv
```

Model selection throughout uses the existing philosophy
(`ml/training/config.py`): **primary metric macro-F1 on the validation
split, tie-break on ESI 1-2 recall**. The test split is only ever scored
once, after selection - never used to choose a model.

## MIMIC-IV-ED schema verification (done this pass)

Before trusting `ml/data/mimic.py`'s column mapping, it was checked against
two real sources rather than assumed:

- **PhysioNet's MIMIC-IV-ED project page** (`physionet.org/content/mimic-iv-ed/2.2/`,
  resolved successfully): confirms the `triage` table's columns as
  `subject_id, stay_id, temperature, heartrate, resprate, o2sat, sbp, dbp,
  pain, acuity, chiefcomplaint` - matching `ml/data/mimic.py`'s
  `_TRIAGE_MAP` exactly - and states "1 indicates the highest severity and
  5 indicates the lowest severity" for `acuity`, matching this repo's ESI
  convention (1 = Resuscitation ... 5 = Non-Urgent).
- A secondary community documentation source
  (`MIT-LCP/mimic-iv-website`'s `content/ed/triage.md`, fetched directly)
  disagrees on two points: it lists different column names (`temp`, `HR`,
  `RR`, `SaO2`, `Pain`, `Acuity`, `SBP`, `DBP`, no `chiefcomplaint` at all)
  and describes `Acuity` in the **opposite** direction ("1 - Not urgent, 5 -
  must be seen immediately"). That page's own "Table purpose"/"Number of
  rows" fields are blank, its `mimic.mit.edu/docs/iv/modules/ed/triage/`
  live URL 404s, and its casing matches no other MIMIC table - all
  consistent with it being a stale/incomplete draft rather than the current
  schema. The PhysioNet page and the near-universal convention in
  published MIMIC-IV-ED research (ESI 1 = most acute) were trusted instead;
  **no code was changed** on the strength of the contradicting source.
- **Not independently confirmed**: temperature's unit (Fahrenheit vs.
  Celsius) in the real `triage.csv`. The PhysioNet page doesn't state it;
  the (likely stale) community page claims Celsius, contradicting the
  adapter's existing Fahrenheit→Celsius conversion and the "MIMIC records
  Fahrenheit" comment already in the code. **This is a real open question**,
  left exactly as implemented (not weakened by an unreliable secondary
  source) but flagged here explicitly: **before training on a real MIMIC-IV-ED
  download, check the raw `temperature` column's value range** - values
  clustering around 97-99 confirm Fahrenheit (conversion needed, as coded);
  values clustering around 36-38 would mean the conversion is wrong and
  needs removing. `ml.config.FEATURE_RANGES["temperature"] = (30.0, 45.0)`
  (Celsius) would clip/flag an un-converted Fahrenheit column as
  out-of-range, which is a partial safety net but not a substitute for
  checking directly.

## Experiment A: synthetic baseline

`Synthetic train -> model -> synthetic test`. Establishes the baseline.
Reuses `ml.training.train.run_training_experiment` exactly as the
production CLI does, just without writing an artifact to `ml/models/`.

**Status: DONE.** `ml/experiments/results/synthetic_baseline.json`
(seed 42, n=12000): xgboost + TF-IDF selected, test accuracy 0.792, macro F1
0.738, weighted F1 0.768, ESI 1-2 recall 0.846 / precision 0.943. Also
reports a basic top-label calibration check (`calibration.ece`).

## Experiment B: synthetic → real (domain-shift measurement)

`Synthetic train -> trained model -> real test, no retraining.` Measures
how much performance drops when a synthetic-trained model sees real ED
patients unmodified. `run_synthetic_to_real(synthetic_df, source, data_path, seed=...)`.

**Status: NOT RUN for either source** - no MIMIC-IV-ED download and no
Indian dataset are available locally (see `RESEARCH_DOCUMENTATION.md`).
`results/synthetic_to_mimic.json` and `results/synthetic_to_indian.json`
each record their own `NOT_RUN` reason rather than a number. Unit-tested
against stand-in data for *both* sources
(`ml/tests/test_experiments.py::test_synthetic_to_real_runs_against_mimic_shaped_data`
confirms the real `ml.data.mimic` column mapping/unit-conversion path
actually executes, not just a renamed copy of the Indian path).

## Experiment C: real-only

`Real train -> model -> real test`, same train/val/test philosophy as
Experiment A, using a patient-level split (`ml.training.splits.patient_level_split`)
automatically if the dataset has a usable patient identifier.
`run_real_only(source, data_path, seed=...)`.

**Status: NOT RUN for either source** - no dataset available.

## Experiment D: combined

`Synthetic train + real train -> model -> real test.` Does adding synthetic
volume help when real data is scarce, evaluated only on real data.

**Status: NOT RUN for either source** - no dataset available. The mechanism
(concatenate train partitions, hold out a real-only test split, fit the full
zoo, select by test macro-F1 *for this demonstration experiment only* - see
the note field in `run_combined()`) is implemented and exercised in tests
with stand-in data.

## Experiment E: adaptation

Method actually implemented: **combined training with real-dataset rows
given a higher `sample_weight`** during a normal `.fit()` call
(`ml.experiments.run_domain_experiment.fit_with_source_weights`, default
weight 3x). This is plain example weighting, not XGBoost continued/
incremental training (`xgb_model=` warm-start) - that is a different,
stronger technique this repo does not currently implement or claim.

Other adaptation strategies considered but not implemented this pass:
hyperparameter re-tuning against a real validation split, and XGBoost
continuation proper. Both are listed in "Future work" in
`RESEARCH_DOCUMENTATION.md`.

**Status: NOT RUN for either source** - no dataset available. The weighting
mechanism itself is unit-tested directly
(`test_fit_with_source_weights_mechanism_runs_and_predicts`) to confirm it
fits and predicts correctly; that test does not claim any domain-adaptation
result.

## Structured-vs-text ablation

Not one of the five domain configs above, but the other required experiment:
does `chief_complaint` (TF-IDF) add anything over the structured vitals/
demographics alone? Both variants are the **same model family**, fit and
scored on the **same** synthetic validation split - only the feature set
changes. This reuses the model zoo's existing `val_results` from Experiment
A's run (`ml.training.train`'s own `text_mode="compare"` sweep), so nothing
is retrained twice.

**Status: DONE.** `ml/experiments/results/structured_vs_text.json` (xgboost,
n=2400 validation rows for both variants):

| | Accuracy | Macro F1 | Weighted F1 | ESI 1-2 Recall | ESI 1-2 Precision |
|---|---:|---:|---:|---:|---:|
| Structured only | 0.618 | 0.543 | 0.576 | 0.716 | 0.869 |
| Structured + TF-IDF | 0.772 | 0.709 | 0.743 | 0.856 | 0.914 |

Chief-complaint text clearly adds predictive information beyond the
structured vitals/demographics alone, **on the synthetic dataset**, where the
generator deliberately encodes real signal in complaint red-flag/low-acuity
phrases (see `ml/data/synthetic.py`). Whether this holds on real MIMIC-IV-ED
or Indian complaint text (different vocabulary, clinical shorthand,
code-mixed language) is untested - see Limitations.

## Reports (independent of the five experiments)

`ml.experiments.reports.class_distribution_report` /
`data_quality_report`, run for every dataset that's actually loaded. Output:
`ml/experiments/reports/class_distribution.csv` and
`ml/experiments/reports/data_quality.json`. The real-dataset side of both is
literally the string `"NOT_AVAILABLE"` right now, not an empty table.
