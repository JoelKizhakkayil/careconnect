# MIMIC-IV-ED Demo → Synthetic Data Calibration

## Status: DONE (against the MIMIC-IV-ED **Demo**, n=207 after cleaning)

Research question: can the MIMIC-IV-ED Demo be used, in aggregate, to
calibrate `ml.data.synthetic.generate_synthetic_dataset` so the synthetic
data it produces is statistically closer to a real ED population - and does
that measurably improve model transfer to the real data, without training on
it? Implementation: `ml.data.calibration` (the calibration mechanism) +
`ml.experiments.run_calibration_experiment` (the orchestration). Numbers
below are read directly from `ml/experiments/results/synthetic_calibration_metrics.json`
and `ml/experiments/results/calibration_model_results.json`, not typed in
separately - the same command reproduces them (section "Reproducibility").

**The most important distinction in this document**: `mimic_demo_reference`
is real, read-only MIMIC-IV-ED Demo data. `synthetic_mimic_calibrated` is a
**newly generated synthetic dataset** - no MIMIC row was copied, perturbed,
or memorised into it. Calibration only ever reads *aggregate* statistics
(means, variances, proportions, counts) off `mimic_demo_reference`; see
"What was calibrated" below for the exact list.

## The three datasets

| Label | What it is | Rows (after cleaning) |
|---|---|---:|
| `synthetic_original` | `ml.data.synthetic.generate_synthetic_dataset`, unmodified | 12,000 |
| `mimic_demo_reference` | Real MIMIC-IV-ED Demo (`edstays` + `triage`), read-only | 207 |
| `synthetic_mimic_calibrated` | New synthetic data, generator parameters derived from `mimic_demo_reference`'s aggregate statistics | 12,000 |

`mimic_demo_reference` has no `patients.csv` in the Demo bundle, so `age` is
100% missing (imputed at fit time, same as every other MIMIC run in this
repo - see `docs/EXPERIMENT_PLAN.md`). Its ESI mix is heavily skewed: ESI
1=8.7%, 2=46.9%, 3=43.5%, 4=1.0%, 5=0.0% - a consequence of the Demo's small,
non-representative 207-row sample, not a claim about real ED acuity
distributions.

## What was calibrated, and what was not

`ml.data.calibration.derive_calibration_params` reads only aggregate
statistics off `mimic_demo_reference` (mean, variance, proportions, per-ESI
subgroup counts) - never an individual row - and produces overrides for
`generate_synthetic_dataset`'s existing keyword parameters (all default to
the original hand-tuned constants, so calling the generator with no
overrides is unchanged):

| Parameter | Method | Rationale |
|---|---|---|
| `vitals_by_severity` | Each latent band's vitals shrunk toward the reference dataset's ESI-conditional mean/variance for the ESI class(es) that band maps to (`critical`→ESI1, `high`→ESI2, `moderate`→ESI3, `low`→ESI4+5), weighted by empirical-Bayes shrinkage `alpha = n / (n + 20)` on that subgroup's size | A 2-row subgroup barely moves the original parameter; a 97-row subgroup moves substantially. Preserves variance structure rather than overwriting it (section 8's "do not simply replace every mean" requirement) |
| `severity_p` | Iterative proportional fit (3 passes, 4,000-row trial generations): after each pass, compare the *resulting* ESI mix to the reference ESI mix and adjust each band's probability by that ratio, floored at 1% per band | Approximate, not an analytic inversion of the generator's stochastic score→ESI mapping (chief-complaint terms and noise also feed the score). The floor stops a class from vanishing from a 100-patient demo's absence of examples - see "A construction detail" below |
| `sex_p` | Global empirical-Bayes shrinkage, `alpha = 207/(207+20) ≈ 0.91`, toward the reference's M/F proportions | The generator already treats sex as band-independent, so a global blend is a like-for-like match to that existing assumption |
| `missingness_rates` (per numeric column, plus newly modelling `heart_rate`/`systolic_bp` missingness, which the original generator never modelled) | Same global shrinkage as `sex_p` | Also already band-independent in the generator |

**Explicitly not calibrated** (see `method_notes` in every run's
`synthetic_calibration_metrics.json` for the same text, machine-readable):

- **`arrival_transport`** - the reference dataset only gives a *marginal*
  (unconditional) transport distribution. Blending it into the generator's
  existing *per-band conditional* distribution would overwrite real
  severity-conditioned structure with unrelated information. Measured
  (distance dropped anyway, as a side effect of the ESI-mix recalibration -
  see below) but never a direct calibration target.
- **The chief-complaint phrase bank** - only aggregate token/category
  statistics are compared (`complaint_text_stats`,
  `complaint_band_distribution`). The generator's fixed phrase bank is left
  untouched; every generated complaint is provably a member of that original
  bank (`ml/tests/test_calibration.py::test_calibrated_complaints_only_come_from_the_generators_own_vocabulary`).
  This is a governance line, not a robustness shortcut: with 207 rows,
  deriving new phrases from the reference text risks reproducing patient-level
  complaint text almost verbatim.
- **`age`** - 100% missing in this Demo extract (no `patients.csv`). That is
  a demo-bundle limitation, not a real clinical missingness signal, so it is
  not something to reproduce.

### A construction detail worth being explicit about

The 207-row Demo has exactly **2** rows at ESI 4 and **0** at ESI 5. Fit
literally, the iterative proportional-fitting step drives the `low`
severity-band probability toward zero, and a 12,000-row draw at that
probability produced only 2 ESI-5 rows - too few for the standard 3-way
stratified train/val/test split used everywhere else in this repo
(`ml.training.splits.stratified_split`). `MIN_BAND_P = 0.01` floors every
band's probability at 1% before renormalizing. This is a deliberate,
documented modelling choice: "zero examples of ESI-5 in a 100-patient demo"
is not strong evidence that ESI-5 has zero true probability, and a floor
keeps every class in the calibrated dataset at least minimally representable
and splittable. With the floor, the calibrated dataset landed at ESI-4=4.9%,
ESI-5=0.7% (vs. the Demo's 1.0%/0.0%) - see the ESI table below.

## Distance metrics: did calibration reduce statistical distance?

`ml.data.calibration.distance_metrics` + `summarize_calibration_improvement`
(section 17's full bundle: KS statistic, Wasserstein distance and mean
difference for numeric features; total-variation distance and Jensen-Shannon
divergence for categorical features and the ESI distribution; mean absolute
difference between correlation matrices). Full detail in
`ml/experiments/results/synthetic_calibration_metrics.json`.

**Overall: 21 of 28 comparable metrics moved closer to `mimic_demo_reference`
(75%).** This is not uniform improvement - reported honestly below, not
rounded up.

### ESI class distribution - the clearest win

| ESI | `mimic_demo_reference` | `synthetic_original` | `synthetic_mimic_calibrated` |
|---|---:|---:|---:|
| 1 | 8.7% | 7.5% | 8.5% |
| 2 | 46.9% | 17.7% | 45.5% |
| 3 | 43.5% | 39.9% | 41.3% |
| 4 | 1.0% | 14.8% | 4.9% |
| 5 | 0.0% | 20.0% | 0.7% |

Mean absolute class-proportion difference: **0.136 → 0.019** (a ~7x
reduction). Jensen-Shannon divergence: **0.203 → 0.014**. See
`docs/figures/esi_distribution.png`.

### Numeric features - mixed by metric, not uniformly better

| Feature | KS statistic (before→after) | Wasserstein distance (before→after) | Abs. mean diff (before→after) |
|---|---:|---:|---:|
| heart_rate | 0.095 → **0.082** ✓ | 1.70 → 2.17 ✗ | 0.41 → 1.80 ✗ |
| respiratory_rate | 0.218 → 0.282 ✗ | 1.41 → **1.10** ✓ | 0.44 → 0.59 ✗ |
| systolic_bp | 0.198 → **0.050** ✓ | 9.21 → **2.88** ✓ | 8.48 → **2.11** ✓ |
| diastolic_bp | 0.274 → **0.129** ✓ | 11.20 → 15.44 ✗ | 2.62 → **1.16** ✓ |
| oxygen_saturation | 0.301 → **0.208** ✓ | 1.43 → **0.84** ✓ | 1.42 → **0.83** ✓ |
| temperature | 0.361 → **0.246** ✓ | 0.58 → 1.44 ✗ | 0.56 → **0.14** ✓ |
| pain_level | 0.342 → **0.286** ✓ | 1.59 → **1.06** ✓ | 0.12 → 0.44 ✗ |

By KS statistic (the metric least sensitive to a handful of outliers, which
matters with only 207 reference rows): **6 of 7 features improved**;
`respiratory_rate` got worse. By Wasserstein distance: 4 of 7 improved,
`heart_rate`/`diastolic_bp`/`temperature` got worse. By raw mean difference:
4 of 7 improved. **`systolic_bp` and `oxygen_saturation` improved on every
metric; `respiratory_rate` improved on none.** See
`docs/figures/numeric_distributions.png` and
`docs/figures/calibration_improvement.png`.

### Categorical features

| Feature | Total variation distance (before→after) | Jensen-Shannon divergence (before→after) |
|---|---:|---:|
| sex | 0.044 → **0.007** ✓ | 0.0014 → **0.00004** ✓ |
| arrival_transport | 0.444 → **0.299** ✓ | 0.260 → **0.190** ✓ |

`arrival_transport` improved even though it was **not** a direct calibration
target (see "What was calibrated" above) - a side effect of the severity-band
probabilities shifting toward the reference ESI mix, since each band still
carries its own (uncalibrated) arrival-transport profile. It remains far
from matched (TVD 0.30 is still large) precisely because it was never
directly targeted.

### Correlation structure

Mean absolute difference across 8 tracked pairs (`heart_rate`↔`systolic_bp`,
`heart_rate`↔`respiratory_rate`, `oxygen_saturation`↔`respiratory_rate`, and
each vital/`age`↔`esi`, Pearson for vital-vital pairs, Spearman against the
ordinal ESI target - correlational, not causal): **0.270 → 0.099**, a clear
improvement. This was never a direct calibration target either (see "What
was calibrated") - it moved because the per-band vitals and severity mix
both did. See `docs/figures/correlation_heatmaps.png`.

### Conditional distributions (vitals given ESI) - more informative than unconditional means

Example, `heart_rate` by ESI (`ml/experiments/results/synthetic_calibration_metrics.json` → `conditional_numeric_by_esi`):

| Dataset | ESI 1 mean (n) | ESI 2 mean (n) |
|---|---|---|
| `mimic_demo_reference` | 109.3 (n=12) | 91.5 (n=95) |
| `synthetic_original` | 120.8 (n=900) | 100.6 (n=2,123) |
| `synthetic_mimic_calibrated` | 116.7 (n=986) | **93.0** (n=5,172) |

The well-populated ESI-2 subgroup (n=95 in the reference) shrinks
substantially toward the reference mean (100.6 → 93.0, reference is 91.5);
the thin ESI-1 subgroup (n=12) shrinks much less (120.8 → 116.7) - exactly
the shrinkage-by-subgroup-size design described above, working as intended.

## Model transfer: does calibration help the model, not just the data?

**Important limitation restated**: Experiments 3 and 4 evaluate on
`mimic_demo_reference`, the same dataset used to derive the calibration
parameters. This is **not external validation** - call it what
`ml.experiments.run_calibration_experiment` labels it in the JSON:
`mimic_reference_calibration_transfer_evaluation`. MIMIC was never trained
on, never used to select a model, and never used to iterate the calibrated
generator after the one aggregate-statistics pass that built it - but
calibrating and evaluating against the same small dataset is still a form of
reuse. Genuine external validation needs an untouched held-out MIMIC-IV-ED
(or other) test set never touched during calibration - out of reach with
only the Demo available (207 rows total, none held out - see Limitations).

| Experiment | Train | Test | Model | Accuracy | Macro F1 | Weighted F1 | ESI1-2 Recall | ESI1-2 Precision |
|---|---|---|---|---:|---:|---:|---:|---:|
| 1: `synthetic_original` baseline | `synthetic_original` (train+val) | `synthetic_original` (held-out) | xgboost+text | 0.792 | 0.738 | 0.768 | 0.846 | 0.943 |
| 2: `synthetic_mimic_calibrated` baseline | `synthetic_mimic_calibrated` (train+val) | `synthetic_mimic_calibrated` (held-out) | xgboost+text | 0.790 | 0.626 | 0.774 | 0.824 | 0.913 |
| 3: `synthetic_original` → `mimic_demo_reference` (no retrain) | `synthetic_original` (train+val) | `mimic_demo_reference` (n=207) | xgboost+text | 0.444 | 0.260 | 0.492 | 0.383 | 0.800 |
| 4: `synthetic_mimic_calibrated` → `mimic_demo_reference` (no retrain) | `synthetic_mimic_calibrated` (train+val) | `mimic_demo_reference` (n=207) | xgboost+text | 0.527 | 0.257 | 0.474 | 0.383 | 0.830 |

**Calibration did not improve macro-F1 transfer, and did not change
high-acuity recall at all** (0.383 in both experiment 3 and 4 - identical).
Accuracy and weighted F1 moved in different directions (accuracy up 0.444 →
0.527, weighted F1 down 0.492 → 0.474), which is itself informative: the
calibrated model got noticeably better at the majority classes (ESI 2/3, now
correctly weighted much closer to their true ~47%/44% share) without getting
better - or worse - at catching the rare high-acuity cases specifically.
Experiment 2's own macro-F1 (0.626, notably lower than Experiment 1's 0.738)
shows the calibrated dataset is also a *harder* dataset to fit well
internally, consistent with its much more skewed class balance (5,380 of
12,000 rows are ESI 2).

**Bottom line: distributional calibration worked; transfer-performance
calibration did not, on this experiment.** Getting the aggregate statistics
closer to MIMIC did not translate into a better transfer model here. With
only 207 reference rows for both calibrating and evaluating, this could be
noise, a genuine ceiling from the danger-score/complaint mechanism being
otherwise unchanged, or both - not distinguishable with this sample size.

## Structured-only vs. structured+TF-IDF (section 22)

**On each dataset's own validation split** (not a transfer measurement):

| Dataset | Structured-only Macro F1 | Structured-only ESI1-2 Recall | +TF-IDF Macro F1 | +TF-IDF ESI1-2 Recall | TF-IDF helps? |
|---|---:|---:|---:|---:|:--:|
| `synthetic_original` | 0.543 | 0.716 | 0.709 | 0.856 | Yes |
| `synthetic_mimic_calibrated` | 0.457 | 0.789 | 0.655 | 0.829 | Yes |

TF-IDF clearly helps in-domain for both datasets, consistent with
`docs/EXPERIMENT_PLAN.md`'s existing finding - not surprising, since
calibration never touched the complaint phrase bank (by design).

**On transfer to `mimic_demo_reference`** (fit on train+val, no MIMIC
training, evaluated on MIMIC - `mimic_reference_calibration_transfer_evaluation`):

| Trained on | Structured-only Macro F1 (Acc / ESI1-2 Recall) | +TF-IDF Macro F1 (Acc / ESI1-2 Recall) | TF-IDF helps on transfer? |
|---|---|---|:--:|
| `synthetic_original` | 0.248 (0.444 / 0.322) | 0.260 (0.444 / 0.383) | Yes |
| `synthetic_mimic_calibrated` | **0.305** (0.556 / **0.617**) | 0.257 (0.527 / 0.383) | **No** |

This is the most surprising result in this experiment: **for the calibrated
dataset, adding TF-IDF text actually hurt transfer to MIMIC** (macro F1 0.305
→ 0.257, high-acuity recall 0.617 → 0.383), the opposite of every other
comparison in this document. The structured-only calibrated model reaches
this experiment's best high-acuity recall (0.617) against MIMIC of any
model tested. A plausible explanation, not confirmed: the calibrated
dataset's *vitals* are closer to MIMIC's real signal, so a structured-only
model leans on that improved signal; the *complaint text* was never
calibrated (by design - see above) and MIMIC's real free-text chief
complaints don't share the synthetic phrase bank's vocabulary, so TF-IDF
features fit on the calibrated data may be picking up noise or
dataset-specific artifacts that don't transfer. This is a hypothesis, not a
demonstrated mechanism - flagged as a direction for the "untested" list
below, not a conclusion.

## Tests

```bash
backend/.venv/Scripts/python.exe -m pytest -v -o addopts=""
```

**127 passed, 0 failed, 0 skipped** (104 pre-existing + 23 new in
`ml/tests/test_calibration.py`, none of the pre-existing tests changed
behaviour). New tests cover: calibration parameters are generated and valid
(`severity_p`/`sex_p` sum to 1, `missingness_rates` in [0,1]); ESI values in
generated output stay in {1..5}; generated rows pass
`ml.data.schema.validate_common_schema`; smaller reference subgroups get
proportionally less shrinkage than larger ones; same-seed reproducibility
(`generate_calibrated_dataset` called twice with identical arguments produces
byte-identical output, both the DataFrame and the parameters); a different
seed produces different output; `subject_id`/`stay_id` never appear in
generated output or in the calibration parameters; a reference DataFrame with
injected `diagnosis`/`disposition` columns produces **identical** calibration
parameters to one without them (proves those columns are never read);
every generated `chief_complaint` is provably a member of the generator's
original fixed phrase bank (structural proof no reference text leaks in).

## Reproducibility

```bash
backend/.venv/Scripts/python.exe -m ml.experiments.run_calibration_experiment \
    --data-path "mimic-iv-ed-demo-2.2/mimic-iv-ed-demo-2.2/ed"

backend/.venv/Scripts/python.exe -m ml.experiments.calibration_plots \
    --data-path "mimic-iv-ed-demo-2.2/mimic-iv-ed-demo-2.2/ed"
```

Fixed inputs, all recorded in `synthetic_calibration_metrics.json`'s
`calibration_params.reference_summary`: random seed 42, `n_samples=12000`,
`tuning_n=4000`, `iterations=3`, `shrinkage_k=20.0`. Given the same MIMIC-IV-ED
Demo download and the same command, every number in this document
regenerates identically (verified: `generate_calibrated_dataset` with fixed
arguments is asserted byte-identical across repeated calls in
`ml/tests/test_calibration.py`). Library versions used for this run:
scikit-learn 1.9.0, pandas 3.0.5, numpy 2.5.2, xgboost 3.4.1, scipy 1.18.1,
Python 3.13.5.

Outputs:

```text
ml/data/calibration.py                                (calibration mechanism)
ml/experiments/run_calibration_experiment.py           (orchestration)
ml/experiments/calibration_plots.py                    (figures)
ml/tests/test_calibration.py                           (23 tests)

ml/experiments/results/synthetic_calibration_metrics.json  (distance metrics, before/after, calibration params)
ml/experiments/results/calibration_model_results.json      (experiments 1-4 + text ablations)
ml/experiments/reports/synthetic_vs_mimic.json              (descriptive comparison, original)
ml/experiments/reports/calibrated_synthetic_vs_mimic.json   (descriptive comparison, calibrated)

ml/data/generated/synthetic_original.csv               (not committed - see .gitignore)
ml/data/generated/synthetic_mimic_calibrated.csv        (not committed - see .gitignore)

docs/figures/numeric_distributions.png
docs/figures/esi_distribution.png
docs/figures/correlation_heatmaps.png
docs/figures/calibration_improvement.png
```

`ml/data/generated/` is not committed - it falls under the repo's existing
`ml/data/**` gitignore rule (datasets are never committed, only code and
small aggregate JSON/CSV reports are). The MIMIC-IV-ED Demo download itself
(`mimic-iv-ed-demo-2.2/`) is separately gitignored at the repo root.

## Limitations

- **The MIMIC-IV-ED Demo (n=207) is tiny relative to full MIMIC-IV-ED**
  (hundreds of thousands of ED stays). Every reference-derived statistic
  here - especially per-ESI subgroup vitals (ESI-4 n=2, ESI-5 n=0) - carries
  large sampling uncertainty. The empirical-Bayes shrinkage is designed to
  limit the damage from that, not eliminate it.
- **The Demo is used as a calibration reference, not ground truth.** Its own
  class balance (ESI 2/3 = 90% of rows, ESI 4/5 ≈ 1%) is itself a property of
  this particular 207-patient sample, not necessarily representative of a
  real ED population at large.
- **`synthetic_mimic_calibrated` remains synthetic data end to end.** No
  MIMIC row was copied, perturbed, or memorized - every generated row comes
  from the same hand-built generative model as `synthetic_original`, only
  with recalibrated parameters (see "What was calibrated" and the leakage
  tests above).
- **Experiments 3 and 4 are calibration-transfer evaluations, not external
  validation** - the same 207 rows both calibrated the generator and scored
  transfer. A real held-out test would need a MIMIC-IV-ED subset never seen
  during calibration; the Demo is too small to carve one out meaningfully.
- **Full MIMIC-IV-ED should be used for the eventual genuine held-out
  external evaluation** described in `docs/EXPERIMENT_PLAN.md` /
  `docs/RESEARCH_DOCUMENTATION.md` - this calibration experiment is a
  complementary, smaller-scope piece of that larger research plan, not a
  replacement for it.
- **No Indian dataset exists locally** - unrelated to this experiment, noted
  for completeness since `docs/RESEARCH_DOCUMENTATION.md` tracks that
  question separately. Nothing in this document claims otherwise.
- **`arrival_transport` and the chief-complaint phrase bank were
  deliberately not calibrated** (see "What was calibrated" for why); their
  remaining distance from MIMIC (TVD 0.30 for transport; phrase bank
  untouched by construction) is expected, not a bug.
- **Do not read this as clinical validation.** Nothing here evaluates
  clinical correctness of any ESI assignment - it evaluates statistical
  similarity between two data-generating processes and one model's transfer
  behaviour between them.

## Research interpretation

1. **Did calibration make synthetic data statistically closer to MIMIC?**
   Yes, on 21 of 28 tracked metrics (75%), and decisively so on the ESI
   distribution (mean abs. diff 0.136 → 0.019, a ~7x reduction) and
   correlation structure (mean abs. diff 0.270 → 0.099).
2. **Which features improved?** ESI distribution, correlation structure,
   sex proportions, `systolic_bp` and `oxygen_saturation` (every metric),
   `diastolic_bp`, `temperature`, `pain_level` (by KS statistic and/or mean
   difference), `arrival_transport` (as a side effect, not a direct target).
3. **Which features remained different, or got worse?** `respiratory_rate`
   got worse by every numeric metric tracked. `heart_rate`, `diastolic_bp`
   and `temperature` improved by KS statistic but got worse by Wasserstein
   distance - the two metrics disagree on those three features, which is
   itself a finding (KS is more robust to the handful of extreme values a
   207-row sample can produce; Wasserstein weighs them more).
   `arrival_transport` improved but remains far from matched (TVD 0.30),
   consistent with never being a direct calibration target.
4. **Did model transfer performance improve?** No, essentially unchanged:
   macro F1 0.260 (original) vs. 0.257 (calibrated) transferring to MIMIC -
   within noise at n=207. Accuracy improved (0.444 → 0.527) but weighted F1
   slightly worsened (0.492 → 0.474), so even "improved" is metric-dependent
   here.
5. **Did ESI 1-2 recall change?** No - identical at 0.383 in both
   experiments 3 and 4. The one place recall moved was in the
   structured-only ablation specifically (0.617 for calibrated-trained,
   structured-only, vs. 0.383 for every TF-IDF-including variant) - see
   point 7.
6. **Did TF-IDF remain useful?** Yes, unambiguously, in-domain (both
   datasets, own validation split). On transfer to MIMIC it helped for
   `synthetic_original` (macro F1 0.248 → 0.260) but **hurt** for
   `synthetic_mimic_calibrated` (0.305 → 0.257, recall 0.617 → 0.383) - the
   single most surprising result in this experiment, discussed above with a
   hypothesis (calibrated vitals carry real improved signal; the
   uncalibrated phrase bank may not transfer as well once the structured
   signal is stronger) that is not confirmed here.
7. **What remains untested?**
   - Why TF-IDF specifically hurts transfer for the calibrated-vitals model
     but not the original model (point 6) - not investigated further here.
   - Whether calibrating the chief-complaint phrase bank itself (not just
     measuring aggregate text stats) would change the transfer/TF-IDF
     picture - deliberately out of scope for governance reasons (see "What
     was calibrated").
   - Whether calibrating `arrival_transport` conditionally (which would need
     a much larger reference sample to estimate per-severity-band transport
     proportions reliably) would move its still-large TVD (0.30) further.
   - Whether the ESI-mix-driven correlation/categorical improvements persist
     or strengthen with a larger reference sample (full MIMIC-IV-ED).
   - Any claim of clinical validity or readiness - out of scope for this
     entire research phase, not just this document.
