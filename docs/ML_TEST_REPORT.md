# CARECONNECT ML TRIAGE TEST REPORT

Note on file layout: the task asked for test files under `tests/test_triage_api.py`,
`tests/test_preprocessing.py` and `tests/test_inference.py`. This repo already has
`test_triage_api.py` and `test_preprocessing.py` under `backend/tests/` and `ml/tests/`
respectively (wired up in the root `pytest.ini` as `testpaths = ml/tests backend/tests`),
so those were reused/extended instead of duplicated. Only `ml/tests/test_inference.py`
was new. The test-case JSON lives at `backend/tests/data/triage_test_cases.json`
instead of `tests/data/`, for the same reason.

## 1. Test Environment

- OS: Windows 11 (`Windows-11-10.0.26200-SP0`)
- Python: 3.13.5
- Test framework: pytest 9.1.1, FastAPI's `TestClient` (httpx under the hood)
- Web framework: FastAPI 0.141.1
- ML libraries: scikit-learn 1.9.0, xgboost 3.4.1, shap 0.52.0, pandas 3.0.5, numpy 2.5.2, joblib 1.6.0, pydantic 2.13.4
- Shipped model: `ml/models/triage_model_v1.joblib` (xgboost, uses chief-complaint text via TF-IDF)
- Model loaded successfully: yes. Confirmed two ways — `ml/tests/test_inference.py::test_load_model_loads_the_shipped_artifact` loads it directly, and `backend/tests/test_triage_explainability.py` loads it through the real `/predict` endpoint and gets a SHAP explanation back.

One thing worth flagging up front: most of the automated suite runs against the **heuristic fallback predictor**, not the trained xgboost model. `backend/tests/conftest.py` forces `TRIAGE_MODEL_BACKEND=heuristic` so the API tests are deterministic. Only one test (`test_triage_explainability.py`) swaps in the real model for a single request.

## 2. Test Scope

The tests were mainly focused on checking whether the triage API and ML pipeline work correctly with normal inputs and a few invalid or incomplete inputs. This covers the FastAPI `/api/emergency/triage/*` endpoints, the feature engineering step (numeric/categorical normalization, TF-IDF on the chief complaint), the two predictor implementations (the trained model and the rule-based heuristic), SHAP explanation output, model loading from disk, and the human-override persistence workflow (using an in-memory fake database, not a live Supabase instance).

## 3. Test Cases

11 synthetic vignettes from `backend/tests/data/triage_test_cases.json`, run against `POST /api/emergency/triage/predict` through the heuristic backend. These are not clinical ground truth — they're inputs picked to exercise different parts of the pipeline (high acuity, low acuity, missing values, unknown categories).

| ID | Test Case | Input Condition | Result | Status |
|----|-----------|-----------------|--------|--------|
| 1 | Severe chest pain | Full vitals, hypotension, hypoxia, pain 9 | API returned ESI 1 (Resuscitation), confidence 0.645, probabilities summing to 1.0 | PASS |
| 2 | Mild headache | Full vitals, all normal, pain 3 | API returned ESI 4 (Less Urgent), confidence 0.588 | PASS |
| 3 | High fever and breathing difficulty | Fever 39.6°C, SpO2 90, RR 28 | API returned ESI 2 (Emergent), confidence 0.588 | PASS |
| 4 | Minor ankle injury | Normal vitals, pain 5 | API returned ESI 4 (Less Urgent), confidence 0.588 | PASS |
| 5 | Severe abdominal pain with vomiting | Fever 38.1°C, HR 102, pain 7 | API returned ESI 4 (Less Urgent), confidence 0.588 — see note below | PASS |
| 6 | Stable cough and sore throat | Normal vitals, pain 2 | API returned ESI 5 (Non-Urgent), confidence 0.645 | PASS |
| 7 | Very low blood pressure with breathing difficulty | SBP 70, SpO2 86, HR 135 | API returned ESI 1 (Resuscitation), confidence 0.645 | PASS |
| 8 | Wrist injury after bicycle fall | Normal vitals, pain 7 | API returned ESI 4 (Less Urgent), confidence 0.588 | PASS |
| 9 | Pediatric fever | Age 7, fever 39.8°C, HR 125 | API returned ESI 4 (Less Urgent), confidence 0.588 | PASS |
| 10 | Unknown categorical values | `sex="NONBINARY"`, `arrival_transport="TAXI"` (not in the schema's known value lists) | Request accepted (HTTP 200), values normalized to fallback categories instead of raising an error | PASS |
| 11 | Missing numeric values | `heart_rate` and `respiratory_rate` omitted | Request accepted (HTTP 200), missing values passed through the imputer without error | PASS |

Note on case 5 (abdominal pain): the heuristic predictor scores this as ESI 4, which is lower than what would typically be expected for "severe abdominal pain with vomiting." This is not a crash or a validation error — the heuristic just doesn't have a keyword rule for abdominal pain and the vitals aren't extreme enough to push the score up on their own. This is a real limitation of the rule-based fallback, not something the tests are hiding.

## 4. ML Pipeline Tests

| Check | Result |
|-------|--------|
| FeatureEngineer execution | Runs and returns a DataFrame with the expected columns (`ml/tests/test_preprocessing.py::test_feature_engineer_adds_derived_columns`, `test_full_pipeline_fits_and_predicts`) |
| Numeric preprocessing | Out-of-range numeric values get clipped instead of raising (`test_feature_engineer_clips_out_of_range_to_nan`); missing numeric values are median-imputed (`test_preprocessor_imputes_missing_numeric`) |
| Categorical preprocessing | Unknown `sex`/`arrival_transport` values are normalized to `UNKNOWN`/`unknown` instead of erroring (`test_feature_engineer_normalises_unknown_category`) |
| TF-IDF processing | Adding the chief-complaint text column increases the feature count as expected, and the full pipeline still fits (`test_text_pipeline_produces_more_features_than_structured_only`) |
| Model prediction | Both the heuristic and a locally-trained RandomForest pipeline produce a class in 1–5 (`ml/tests/test_predictor.py`) |
| Probability output | Probabilities cover all 5 ESI classes and sum to ~1.0, for both predictors (`test_heuristic_probabilities_sum_to_one_and_cover_all_classes`, `test_model_predictor_wraps_pipeline`) |
| SHAP explanation | Returns a structured result with `impact` and `importance` per feature (`ml/tests/test_explainer.py::test_explanation_is_structured`); confirmed against the real shipped model through the live API in `backend/tests/test_triage_explainability.py` (`method` came back `"shap"`, not `"none"`) |
| SHAP failure handling | When the underlying model object is broken, `explain_model_prediction` returns `method: "none"` instead of raising (`test_explanation_returns_none_on_bad_model`) |
| Model loading | `load_model` loads the real shipped artifact and returns metadata; raises `ModelUnavailableError` for a missing file and for an artifact without `predict_proba` (`ml/tests/test_inference.py`, new this pass) |
| Missing values | Covered at both the preprocessing level (above) and the API level (case 11 in the table) |
| Unknown categories | Covered at both the preprocessing level (above) and the API level (case 10 in the table) |

## 5. API Tests

Endpoints actually exercised, all through FastAPI's `TestClient` (no separately running server, no live Supabase — a fake in-memory Supabase client from `backend/tests/conftest.py` stands in for the database):

- `GET /api/emergency/triage/health` — 200, reports `status: "ready"` and a model type
- `POST /api/emergency/triage/predict` — 200 for valid input; 422 for an out-of-range vital, too few vitals, an unrecognized extra field, and an empty or missing chief complaint; 200 (not a crash) for values at the exact edges of the allowed vital ranges; 503 when the triage service is disabled
- `POST /api/emergency/triage` — 201, persists a prediction row and optionally records a human final ESI in the same call
- `GET /api/emergency/triage` — 200, returns only the caller's own rows
- `GET /api/emergency/triage/{id}` — 404 for an id that doesn't exist
- `PATCH /api/emergency/triage/{id}` — 200, records the override without changing the stored model prediction, probabilities, confidence, explanation, or model metadata

## 6. Edge Cases

- **Missing numeric values**: `heart_rate` and `respiratory_rate` omitted from the request. Accepted (200), handled by the preprocessing pipeline's imputer.
- **Unknown categorical values**: `sex` and `arrival_transport` set to values outside the schema's known lists (`NONBINARY`, `TAXI`). Accepted (200), normalized to fallback categories (`OTHER` / `other`) rather than rejected or crashing.
- **Empty/missing chief complaint**: tested both an empty string and the field left out entirely. Both come back 422 (Pydantic's `min_length=1` on a required field) — this is a controlled validation error, not a crash.
- **Extreme vital values**: tested values sitting exactly at the schema's allowed min/max (e.g. `heart_rate=300`, `temperature=45.0`, `age=120`). Accepted (200). Values further outside the range (e.g. `heart_rate=900`) are rejected with 422, which was also tested.
- **Invalid input**: an unrecognized extra field in the request body is rejected with 422 (the schema forbids extra fields).
- **Missing/corrupt model file**: tested at the unit level only — `load_model` raises `ModelUnavailableError` for a missing file and for a file that doesn't contain a usable estimator (`ml/tests/test_inference.py`). This was **not** tested by actually corrupting the live model file and hitting the running API — that would require swapping the artifact out from under a running process, which this pass didn't attempt.

## 7. Failures / Issues Found

No functional failures were observed during the tests that were run.

One warning did show up in every run and is worth recording rather than hiding: pytest prints a `StarletteDeprecationWarning` — "Using `httpx` with `starlette.testclient` is deprecated; install `httpx2` instead." — coming from `fastapi/testclient.py`. It doesn't fail any test and doesn't affect the results in this report, but it's a real warning from the installed library versions and will probably need addressing at some point (see Next Steps).

## 8. Test Summary

Total tests: 67
Passed: 67
Failed: 0
Skipped: 0

**Update (research phase - synthetic-to-Indian domain transfer):** 35 more tests
were added under `ml/tests/` (`test_schema.py`, `test_indian_loader.py`,
`test_splits.py`, `test_experiments.py`) covering the common-schema
validator, the generic Indian dataset loader, patient-level splitting, and
the experiment runner. That pass: 102 passed, 0 failed, 0 skipped.

**Update (real-data experiment pass - MIMIC-IV-ED):** the experiment runner
was generalized to take any real-data source (`--source mimic` or
`--source indian`) instead of being Indian-only, and 2 more tests were added
(`test_synthetic_to_real_not_run_for_mimic_when_no_path_given`,
`test_synthetic_to_real_runs_against_mimic_shaped_data` - the latter
exercises the real `ml.data.mimic` column-mapping/unit-conversion path with
a fixture, confirming the generalization isn't just a renamed copy of the
Indian path). Full suite as of that work: **104 passed, 0 failed, 0
skipped** (`backend/.venv/Scripts/python.exe -m pytest -v -o addopts=""`).
See `docs/RESEARCH_DOCUMENTATION.md` and `docs/EXPERIMENT_PLAN.md` for that
work's own report, including a real MIMIC-IV-ED schema-verification
discrepancy found and left unresolved rather than guessed at.

All tests that were written for this pass ran and passed against the current implementation, on the heuristic backend for the API-level tests and the real model for the loading/explainability checks that specifically needed it. This doesn't say anything about clinical accuracy — it says the API accepts the documented inputs, rejects the documented invalid inputs with the right status code, doesn't crash on missing/unknown values, and the override workflow keeps the model's original output and the human's decision separate.

## 9. Limitations

- The shipped model was trained on synthetic data (`ml/data/synthetic.py`), not real patient records.
- Passing these tests says nothing about clinical performance — it only confirms the software behaves correctly (valid responses, no crashes, correct status codes).
- These tests check software and pipeline behavior, not medical correctness of any prediction.
- No evaluation against MIMIC-IV-ED (or any real clinical dataset) was done as part of this pass. `ml/data/mimic.py` exists for that but wasn't run here — there's no dataset available in this environment.
- The 11-scenario table reflects the heuristic fallback, not the trained xgboost model, because that's what the test suite runs deterministically. The real model's predictions on the same 11 cases were not tabulated.

## 10. Next Steps

- Look at why the heuristic under-scores the abdominal pain case (case 5) — either accept it as a known limitation of the rule-based fallback, or add an abdominal-pain keyword to `HeuristicTriagePredictor`'s rule list.
- Run the 11 scenarios against the real trained model too (not just the heuristic) and compare — this would need temporarily overriding `TRIAGE_MODEL_BACKEND` the same way `test_triage_explainability.py` does, for all 11 cases instead of one.
- Address the `StarletteDeprecationWarning` (httpx/starlette testclient) before it turns into an actual breakage in a future FastAPI/Starlette upgrade.
- Add a database integration test against a real (non-fake) Supabase instance if/when test credentials for one become available — everything right now uses the in-memory fake from `conftest.py`.
- Evaluate the pipeline on MIMIC-IV-ED once that dataset is available locally, using the existing `ml/data/mimic.py` loader and `ml/training/evaluate.py`.
