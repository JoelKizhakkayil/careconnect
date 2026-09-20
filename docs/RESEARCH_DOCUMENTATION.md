# Research Documentation: Synthetic-to-Indian Domain Transfer for ED Triage

## 1. Research problem

> Does a multimodal (structured vitals + chief-complaint text) emergency
> triage model, developed and validated on synthetic data, transfer to an
> Indian emergency-care context? Does adapting it with real Indian data
> improve performance while maintaining high-acuity (ESI 1-2) detection?

## 2. Motivation

CareConnect's shipped triage model (`triage_model_v1`) is trained entirely
on synthetic data (`ml/data/synthetic.py`) - useful for building and testing
the pipeline end to end, but it says nothing about real-world performance,
and says nothing at all about performance in an Indian emergency-care
setting specifically, which differs from the U.S. data (MIMIC-IV-ED) the
architecture was originally designed around: different triage protocols in
use (see AIIMS Triage Protocol, References), different patient population
and disease burden, and chief-complaint text very likely in different
language/vocabulary. Before any claim of "Indian-context" utility can be
made, that transfer has to actually be measured, not assumed.

## 3. Related work

- **MIMIC-IV-ED** (PhysioNet): the architecture's original real-data target,
  U.S.-based, ESI-scale triage acuity recorded at triage. `ml/data/mimic.py`
  is a complete, tested adapter, never exercised in this repo (credentialed,
  not committed).
- **EMcounter** (Chennai, Sundaram Medical Foundation): identified as the
  motivating Indian emergency-care research lead. Investigated directly (see
  §4) - it is a **data-collection tool/survey**, not a public dataset.
- **Indian ED triage research**: the AIIMS Triage Protocol (ATP) is a
  distinct, non-ESI Indian triage protocol used at a high-volume Delhi ED
  since 2010 - relevant because it demonstrates that "Indian ED triage"
  cannot be assumed to already use the ESI scale this model predicts. See
  References for the primary sources reviewed.

## 4. Dataset sources

### Synthetic (used today, ships with the model)

`ml/data/synthetic.py::generate_synthetic_dataset`. Hand-built generative
process (latent severity band → vitals + chief complaint → ESI via a
scorecard + noise), **not clinically valid**, exists purely so the full
pipeline (preprocessing → model zoo → selection → SHAP → API) runs end to
end without any external data dependency. This is unchanged by this
research phase and remains the default for development, CI, and the shipped
demo model.

### MIMIC-IV-ED (optional, real, non-Indian)

Complete adapter at `ml/data/mimic.py`, credentialed via PhysioNet, never
committed. Included in this repo before this research phase started; not
Indian data, documented here only to be explicit it isn't being confused
with the Indian-context objective.

**Checked this pass**: the repo (`ml/data/`), common local download
locations, and the session environment were searched for an existing
MIMIC-IV-ED download (`edstays.csv`/`triage.csv` or `.csv.gz`, or an
equivalent directory). **None was found** - only the adapter code exists,
no data. The adapter's column mapping was independently re-verified against
PhysioNet's own MIMIC-IV-ED project page (see `docs/EXPERIMENT_PLAN.md`'s
"MIMIC-IV-ED schema verification" section for exactly what was checked and
one open discrepancy - temperature units - that was found and not silently
resolved). Since MIMIC-IV-ED is credentialed, this repo does not and will
not attempt to download it automatically; a human with a signed PhysioNet
Data Use Agreement must place it locally (`ml/data/README.md`).

### Indian dataset (objective of this research phase)

**Not currently available.** Investigated specifically:

**EMcounter** - the published status report (Bhoi et al., *EMcounter:
charting the epidemiology of medical emergencies in India - a status
report*, PMC2536178, 2008) describes EMcounter as "a standardized,
user-friendly [web-based] data entry tool" forming the backbone of a planned
multi-center surveillance effort, piloted at Sundaram Medical Foundation,
Chennai, intended to expand to 20+ centers. It collects demographics,
pre-hospital transport, chief complaints, vitals, interventions, disposition
and diagnosis - but the paper's own text states the resulting database is
"accessible to all participating institutions, with institutional
identifiers blinded on request" - i.e. shared among participating sites, not
published. No public repository, DOI, or download link is given anywhere in
the article. A follow-up search for any more recent public release or a
downloadable patient-level file (2024-2026) found none. **Conclusion: no
legitimate publicly downloadable patient-level EMcounter dataset exists.**
This repo does not fabricate one.

**Other Indian ED sources checked**: a search for a public AIIMS or other
Indian-hospital ED triage dataset likewise found protocol descriptions and
published summary statistics, but no downloadable patient-level dataset or
repository.

**What this means for the codebase**: `ml/data/indian.py` implements the
loader interface, validation, and a documented example column-mapping
template, ready for the moment a legitimate dataset is obtained (e.g.
through a direct data-sharing agreement with a participating EMcounter site,
an IRB-approved local ED registry export, or a future public release) - see
§13 for exactly what "ready" means here, and §12 for what must NOT be done
in the meantime.

## 5. Data schema

`docs/DATA_DICTIONARY.md` - the common schema every loader returns, and the
exact source-column mapping for each of the three loaders.

## 6. Preprocessing

Unchanged - `ml.preprocessing.pipeline.FeatureEngineer` (clip out-of-range
numerics to NaN, derive `shock_index`/`pulse_pressure`, normalize
categoricals to known vocabularies, lower-case text) → `ColumnTransformer`
(median/mode imputation, scaling, one-hot, TF-IDF) → estimator, fit only on
the training split (`ml/preprocessing/pipeline.py`). This research phase
does not change preprocessing; the same pipeline runs for every data source.

## 7. Leakage prevention

`ml.config.INITIAL_TRIAGE_FEATURES` is the formal, named whitelist (an alias
of the pre-existing `ALL_INPUT_FEATURES`) - `ml.training.train` selects
exactly these columns before fitting, regardless of what else a loader's raw
source file contains. `ml.config.FORBIDDEN_LEAKAGE_COLUMNS` names
post-triage columns (diagnosis, disposition, ICU admission, mortality,
length of stay, medications, labs, notes) that must never reach the model;
`ml.data.indian.load_indian_dataset` explicitly logs a warning if any are
present in a raw source file and excludes them by construction (only
`column_map`'s named source columns are ever selected -
`ml.data.schema.to_common_schema`). Full per-column "why this is safe at
triage time" rationale (originally written for MIMIC-IV-ED, applies
identically to any ED dataset): `ml/features.md`.

## 8. Experimental design

`docs/EXPERIMENT_PLAN.md` - five domain-transfer configurations (synthetic
baseline, synthetic→real, real-only, combined, adapted) plus the
structured-vs-text ablation, what each measures, and their actual run
status. "Real" is generic on purpose: `ml.experiments.run_domain_experiment`
takes `--source {mimic,indian}`, so the same four configurations answer
"does this transfer to MIMIC-IV-ED?" (checked this pass - not locally
available) and "does this transfer to an Indian dataset?" (still pending
access) without separate code paths. `docs/EXPERIMENT_PLAN.md` also records
the MIMIC-IV-ED schema verification actually performed this pass, including
one genuine open discrepancy (temperature units) that was not silently
resolved.

## 9. Metrics

`ml.training.evaluate.evaluate_predictions` (pre-existing, reused unchanged):
accuracy, macro precision/recall/F1, weighted F1, per-class precision/
recall/F1/support, full confusion matrix, and ESI 1-2 ("high acuity")
precision/recall/F1 reported separately. New this phase:
`ml.training.calibration.expected_calibration_error` - a basic top-label
expected calibration error, computed when probability outputs are available
(currently only reported for Experiment A). Model selection never uses the
test set (`macro_f1`, tie-break `high_acuity.recall`, on the **validation**
split only) - the one exception, noted explicitly in its own output, is
Experiment D (`combined`), a small demonstration split that selects by test
macro-F1 purely for illustration and says so.

## 10. Domain-shift methodology

`docs/DOMAIN_SHIFT_ANALYSIS.md` - `ml.experiments.domain_shift.compare_distributions`
compares numeric distributions, categorical distributions, chief-complaint
vocabulary overlap, and target (ESI) distribution between any two datasets
in the common schema. Mechanism implemented and tested; the actual
synthetic-vs-MIMIC and synthetic-vs-Indian comparisons are both **TBD**,
pending dataset access.

## 11. Adaptation methodology

Implemented: combined training with real-dataset rows sample-weighted higher
(`ml.experiments.run_domain_experiment.fit_with_source_weights`) - ordinary
`sample_weight` during `.fit()`, explicitly *not* called "fine-tuning" and
explicitly not XGBoost continued/incremental training (`xgb_model=`
warm-start), which is a different technique this repo does not implement.
See `docs/EXPERIMENT_PLAN.md` §Experiment E for what was considered and not
implemented (hyperparameter re-tuning on a real validation split; true
XGBoost continuation).

## 12. Results

| Experiment | Train | Test | Accuracy | Macro F1 | ESI 1-2 Recall | Status |
|---|---|---|---:|---:|---:|---|
| Synthetic baseline | synthetic | synthetic | 0.792 | 0.738 | 0.846 | **DONE** |
| Synthetic → MIMIC-IV-ED | synthetic | MIMIC | - | - | - | NOT RUN: no local MIMIC-IV-ED download |
| MIMIC-only | MIMIC | MIMIC | - | - | - | NOT RUN: no local MIMIC-IV-ED download |
| Synthetic → Indian | synthetic | Indian | - | - | - | NOT RUN: no Indian dataset available |
| Indian-only | Indian | Indian | - | - | - | NOT RUN: no Indian dataset available |
| Combined | synthetic+real | real | - | - | - | NOT RUN: no MIMIC or Indian dataset available |
| Adapted (weighted) | synthetic+real (weighted) | real | - | - | - | NOT RUN: no MIMIC or Indian dataset available |

Structured-vs-text ablation (synthetic, same validation split, xgboost),
**DONE**:

| Feature set | Accuracy | Macro F1 | ESI 1-2 Recall | ESI 1-2 Precision |
|---|---:|---:|---:|---:|
| Structured only | 0.618 | 0.543 | 0.716 | 0.869 |
| Structured + TF-IDF | 0.772 | 0.709 | 0.856 | 0.914 |

All source numbers: `ml/experiments/results/*.json`,
`ml/experiments/reports/*.csv`.

**The transfer question this research phase set out to answer - whether the
synthetic-trained model transfers to real emergency-department data (MIMIC-IV-ED
or Indian), and whether real data improves it - remains untested.** A local
MIMIC-IV-ED download was actively checked for and is not present (see §4);
no Indian dataset exists either. Only the synthetic baseline and the
(synthetic-only) text ablation actually ran.

## 13. Limitations

- No Indian patient-level dataset was available at the time of this work;
  every Indian-dependent experiment reports `NOT_RUN`, not a placeholder
  number.
- The synthetic dataset is not clinically valid and was never claimed to be;
  its text-signal advantage (§12) may not generalize to real complaint text
  in any language/vocabulary, Indian or otherwise.
- No calibration, fairness, or subgroup analysis beyond the basic ECE check
  on the synthetic baseline.
- `ml/data/indian.py`'s example column mapping is a template, not validated
  against any real dataset's actual header.
- Patient-level splitting (`ml.training.splits.patient_level_split`) is
  implemented and unit-tested with a synthetic multi-visit fixture, but has
  never been exercised against a real dataset's actual patient identifier
  conventions.

## 14. Future work

0. Obtain a local MIMIC-IV-ED download (PhysioNet credentialing + signed Data
   Use Agreement - see `ml/data/README.md`) and run
   `python -m ml.experiments.run_domain_experiment --experiment all --source mimic --data-path <path>`.
   Before trusting the result, check the raw `temperature` column's value
   range against the open question in `docs/EXPERIMENT_PLAN.md` (Fahrenheit
   vs. Celsius) - this affects whether `ml/data/mimic.py`'s unit conversion
   is correct for that specific downloaded copy.
1. Obtain a legitimate Indian ED dataset - most plausibly through a direct
   data-sharing agreement with a participating EMcounter site or hospital
   ED, or a future public release, following institutional/IRB requirements.
2. Run Experiments B-E once either dataset exists; this requires no further
   architecture work, only `--source` and `--data-path`.
3. If the dataset uses a non-ESI triage scale (plausible - see AIIMS Triage
   Protocol), get a clinically-reviewed `esi_mapping` before using it for
   anything beyond a labeled sensitivity analysis.
4. Extend the adaptation experiment to try hyperparameter re-tuning on an
   Indian validation split, and true XGBoost continued training, and compare
   both against the sample-weighting result actually implemented.
5. Re-run `ml/experiments/domain_shift.py`'s comparison for real once Indian
   data exists, and specifically inspect chief-complaint vocabulary overlap,
   since the structured-vs-text ablation's synthetic result may not
   generalize.

## 15. References

- MIMIC-IV-ED, PhysioNet: <https://physionet.org/content/mimic-iv-ed/2.2/>
- MIMIC-IV-ED triage module documentation: <https://mimic.mit.edu/docs/iv/modules/ed/triage/>
- Bhoi S, et al. "EMcounter - charting the epidemiology of medical
  emergencies in India: a status report." *Int J Emerg Med* (2008). PMC:
  <https://pmc.ncbi.nlm.nih.gov/articles/PMC2536178/>
- Indian ESI-related research: <https://www.jept.ir/index.php/jept/article_91815.html>
- AIIMS Triage Protocol research: <https://pmc.ncbi.nlm.nih.gov/articles/PMC7472824/>, <https://pmc.ncbi.nlm.nih.gov/articles/PMC9639733/>
- AIIMS Triage Protocol (ATP) document: <https://acee-india.org/downloads/atp_aiims.pdf>

## What this project does and does not claim

**Does claim** (contribution of the system as it stands): multimodal
structured + text triage prediction; explainable output (SHAP, with a
heuristic fallback that explains itself the same way); human-in-the-loop
clinician override with the model's original prediction preserved
unmutated; auditability of model vs. human decision
(`triage_predictions` table); a controlled, documented initial-triage
feature boundary (§7); and, as of this phase, an architecture and
methodology for investigating synthetic-to-Indian domain transfer and
domain adaptation.

**Does not claim**: a new ML algorithm; clinical validation of any kind;
superior clinical performance; deployment readiness; or Indian validation -
that investigation is built and ready, but, per §12, has not yet been run
against real data.
