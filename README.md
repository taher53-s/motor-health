# Motor Health: Bearing Condition Monitoring

**Academic project · BTECH CAP 501 · ATLAS / uGDX**

**Team:** Taher Sohagpurwala, Harsh Agrawal, and Manthan Kumar

[Live dashboard](https://motor-health.vercel.app) · [Dataset](evidence/data/) · [Evaluation report](evidence/report.json) · [Presentation](presentation/Motor_Health_8_Slides_Enhanced.pptx)

## Project overview

A project life cycle management assignment with a deployed JavaScript inference API, a continuous dashboard that replays recorded vibration windows, and reproducible Python evaluation.

## What the dashboard shows

The dashboard uses eight held-out recordings from the [CWRU Bearing Data Center](https://engineering.case.edu/bearingdatacenter/48k-drive-end-bearing-fault-data), 48 kHz drive-end channel, at 2 and 3 HP test loads. It updates about once per second by sending the next recorded feature window to the backend. **It is recording playback, not a connected motor or field telemetry.** Names such as `IR007_2` identify CWRU fault type and load; they are not installed assets. HP is horsepower of the test rig load.

Each 2,048-value acceleration window yields RMS, kurtosis and crest factor. The selected RMS rule scores the window against a validation-set threshold; Isolation Forest and an autoencoder are comparison models. The displayed 0–1 health index is a deterministic transform of score/threshold, **not a probability of failure**. Green is at least 0.90, watch is 0.75 to below 0.90, and inspect is below 0.75.

## Reproduce the evaluation

Install `evidence/requirements.txt`, then run `python evidence/run_experiment.py`. The script downloads the 13 official CWRU `.mat` files and verifies their SHA-256 values against `evidence/expected_manifest.json`. It keeps entire recordings in train, validation or test partitions. All 13 original MATLAB recordings are included in `evidence/data/`, with SHA-256 checksums in `evidence/expected_manifest.json`. Processed features are in `evidence/features.csv` and model predictions are in `evidence/predictions.csv`. The generated outputs are in `evidence/results/`; the checked-in release artifacts are in `evidence/`.

The held-out set has 1,897 windows from eight recordings. Windows within the same recording are correlated. RMS and autoencoder each reach AUC 1.000 on this controlled set, which does not establish perfect field performance or early fault prediction. `evidence/parity-tests.json` records 8,889 Python/Worker score comparisons (2,963 windows × 3 models); these verify implementation agreement, not accuracy. NASA IMS was identified during planning but not evaluated in this experiment.

## Run and inspect

`build_release.py` creates a self-contained Cloudflare Worker at `dist/server/index.js` from the checked-in artifacts and UI. `test.mjs` exercises the API and playback scores. The key endpoints are `GET /api/health`, `GET /api/report`, `GET /api/samples`, `GET /api/replay?recording=99&window=0`, and `POST /api/predict` for a 2,048-value 48 kHz CWRU-scale window.

The `presentation/` directory contains the eight-slide project presentation. Planned extensions described there should be distinguished from the implemented functionality documented here.

## Vercel deployment and backend storage

Run `npm run build` and `npm test`, then `vercel --prod`. The TypeScript entry point in `api/index.ts` adapts the existing Fetch handler to a Vercel Node.js Function. All application routes retain their existing behavior. No npm runtime dependencies or secrets are required.

The backend runs on Vercel. Model parameters, recordings, and evaluation artifacts are versioned in Git and bundled into each deployment. `evidence/evidence.sqlite` is a read-only evaluation snapshot offered as a download, not a writable production database. Prediction requests run in memory and are not saved. No Supabase or other external database is configured. Persistent users, sensor readings, and prediction history would require a separate database integration.



## Local setup

Requirements: Node.js 24 and Python 3. Python evaluation dependencies are listed in `evidence/requirements.txt`.

```bash
npm run build
npm test
npm run dev
```

Open `http://localhost:3000`. To reproduce the scientific evaluation:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r evidence/requirements.txt
python3 evidence/run_experiment.py
```

## Submission contents

| Location | Purpose |
| --- | --- |
| `index.html` | Dashboard interface |
| `inference.mjs` | Feature extraction and model scoring |
| `api/index.ts` | Vercel backend entry point |
| `evidence/data/` | 13 original MATLAB recordings |
| `evidence/features.csv` | Extracted vibration features |
| `evidence/predictions.csv` | Evaluation predictions |
| `evidence/evidence.sqlite` | Read-only evaluation database |
| `evidence/expected_manifest.json` | Source URLs and verification checksums |
| `evidence/report.json` | Evaluation metrics and experiment details |
| `presentation/` | Project presentation |

## Scope and future work

This submission demonstrates reproducible benchmark evaluation, backend inference, and an interactive recording dashboard. Physical sensor integration, persistent telemetry storage, automated fault notifications, and validation on industrial equipment are future work.

## Data acknowledgement

The vibration recordings originate from the Case Western Reserve University Bearing Data Center. Source URLs and checksums are retained in the dataset manifest. The recordings are third-party research data; this repository does not claim authorship or ownership of them.
