# Data Dictionary

The single schema every CareConnect data loader (`ml/data/synthetic.py`,
`ml/data/mimic.py`, `ml/data/indian.py`) must return, and how each source's
own columns map onto it. Enforced in code by `ml.config.INITIAL_TRIAGE_FEATURES`
and `ml.data.schema.validate_common_schema` - this document explains it,
the code enforces it.

## Common schema

| Column | Type | Range / values | Notes |
|---|---|---|---|
| `age` | numeric | 0-120 | |
| `sex` | categorical | `M`, `F`, `OTHER`, `UNKNOWN` | Unrecognized source values normalize to `UNKNOWN`, never dropped. |
| `arrival_transport` | categorical | `ambulance`, `walk_in`, `public`, `private`, `helicopter`, `police`, `other`, `unknown` | Unrecognized source values normalize to `other`. |
| `heart_rate` | numeric | 10-300 bpm | |
| `respiratory_rate` | numeric | 2-80 /min | |
| `systolic_bp` | numeric | 40-300 mmHg | |
| `diastolic_bp` | numeric | 20-200 mmHg | |
| `oxygen_saturation` | numeric | 50-100 % | |
| `temperature` | numeric | 30.0-45.0 °C | Always Celsius in the common schema - source-specific unit conversion happens in the loader (see MIMIC below). |
| `pain_level` | numeric | 0-10 | |
| `chief_complaint` | free text | - | Preserved as-is (lower-cased by the preprocessing pipeline, not by loaders). Missing is left as `NaN`/empty, never fabricated. |
| `esi` (target) | int | 1-5 | Emergency Severity Index, 1 = most acute. Never silently remapped from another triage scale - see "Triage scale" below. |

Derived (computed by `FeatureEngineer`, not by any loader): `shock_index =
heart_rate / systolic_bp`, `pulse_pressure = systolic_bp - diastolic_bp`.

Full ranges: `ml/config.py::FEATURE_RANGES`. Full leakage rationale for every
column: `ml/features.md`.

## Per-source mapping

### Synthetic (`ml/data/synthetic.py`)

Generates the common schema directly - there is no external source columns
to map. Not clinically valid; a hand-built scorecard (vitals + complaint
red-flags + noise -> ESI), documented in the module docstring.

### MIMIC-IV-ED (`ml/data/mimic.py`)

| MIMIC-IV-ED column | Table | CareConnect column | Handling |
|---|---|---|---|
| `heartrate` | `triage` | `heart_rate` | direct |
| `resprate` | `triage` | `respiratory_rate` | direct |
| `sbp` | `triage` | `systolic_bp` | direct |
| `dbp` | `triage` | `diastolic_bp` | direct |
| `o2sat` | `triage` | `oxygen_saturation` | direct |
| `temperature` | `triage` | `temperature` | °F → °C conversion |
| `pain` | `triage` | `pain_level` | free text ("7", "denies", "8/10") → numeric via regex extraction, clipped 0-10 |
| `chiefcomplaint` | `triage` | `chief_complaint` | direct |
| `acuity` | `triage` | `esi` (target) | rows with acuity outside 1-5 are dropped, counted, and logged |
| `gender` | `edstays` | `sex` | uppercased |
| `arrival_transport` | `edstays` | `arrival_transport` | lowercased, spaces → underscores |
| `anchor_age` | `patients` (optional join) | `age` | only available if a `patients` table copy is supplied; otherwise `age` is all-`NaN` (imputed) and this is logged |

Credentialed (PhysioNet Data Use Agreement) - never committed. See
`ml/data/README.md` for the access procedure and exact directory layout
expected by `--data-dir`.

### Indian dataset (`ml/data/indian.py`)

No real Indian ED dataset is bundled with or downloaded by this repo (see
`RESEARCH_DOCUMENTATION.md` for what was actually checked). The loader takes
a `column_map` argument - `{source_column_name: careconnect_column_name}` -
supplied by whoever points it at a real file, because no specific dataset's
column names are known yet. `EXAMPLE_COLUMN_MAP` in that file is a template,
not a claim about any real dataset's actual header.

**Triage scale**: if the real dataset does not use ESI 1-5 (for example,
AIIMS' own protocol - see References - uses a different category scheme,
not ESI), the loader will **not** guess a conversion. Pass an explicit
`esi_mapping={"source_value": esi_int, ...}` to `load_indian_dataset`, and
document why that mapping is clinically reasonable before using it for
anything beyond a labeled sensitivity check.

## References

- MIMIC-IV-ED triage module documentation: <https://mimic.mit.edu/docs/iv/modules/ed/triage/>
- MIMIC-IV-ED on PhysioNet: <https://physionet.org/content/mimic-iv-ed/2.2/>
- AIIMS Triage Protocol (a different, non-ESI Indian ED triage scheme, useful context for why ESI cannot be assumed): <https://acee-india.org/downloads/atp_aiims.pdf>
