# 07. Operations and command contract

## 1. Current status

**All implementation commands below are planned interfaces.** The package, environment, datasets and artifacts do not exist yet. T01 onward must implement these interfaces and update this document with tested commands. Do not run the sequence blindly before its prerequisites exist.

Use PowerShell from `04_FraudGuard_Transaction_Fraud_Detection`. Check local Python availability first. Routine execution never modifies neighbouring projects.

## 2. Environment setup at T01

```powershell
Get-Command py, python, git, docker -ErrorAction SilentlyContinue
py -3.11 -m venv .venv
$fgPython = (Resolve-Path '.venv\Scripts\python.exe').Path
& $fgPython -m pip install --upgrade pip
& $fgPython -m pip install -e '.[dev]'
& $fgPython -m pip freeze > requirements-lock.txt
& $fgPython -m fraudguard.cli --help
```

If `py -3.11` is unavailable, choose a verified compatible installed interpreter and record its exact path/version before creating the environment. Do not install a global Python runtime merely because this example names one.

On later sessions, set `$fgPython` again. Exact pinned dependencies belong in `requirements-lock.txt`; record dependency validation in the progress log.

## 3. Acquire and validate data

The default is user-authorized/manual Kaggle acquisition after T02 confirms access and terms. Place the original file at `data/raw/train_transaction.csv`. Do not put credentials in commands, logs or Git.

```powershell
& $fgPython -m fraudguard.cli validate --input data/raw/train_transaction.csv --config configs/project.json --report reports/data_quality.json
& $fgPython -m fraudguard.cli split --input data/raw/train_transaction.csv --config configs/project.json --output-dir data/processed
```

Validation selects the allowlist from the wider raw file. Splitting preserves chronological ties and creates immutable manifests. CLI scoring uses the stricter unlabelled schema.

## 4. Train, select and freeze

```powershell
& $fgPython -m fraudguard.cli train --models prior amount logistic --config configs/project.json --split-manifest data/processed/split_manifest.json --output-dir artifacts/baselines
& $fgPython -m fraudguard.cli train --models lightgbm --config configs/project.json --split-manifest data/processed/split_manifest.json --output-dir artifacts/challengers
& $fgPython -m fraudguard.cli tune --config configs/project.json --split-manifest data/processed/split_manifest.json --output-dir artifacts/selection
& $fgPython -m fraudguard.cli calibrate --selection artifacts/selection/selected.json --baseline-dir artifacts/baselines --config configs/project.json --output-dir artifacts/champion
& $fgPython -m fraudguard.cli freeze-policy --artifact-dir artifacts/champion --baseline-dir artifacts/baselines --split-manifest data/processed/split_manifest.json --review-fraction 0.01 --output reports/freeze_manifest.json
```

- `amount` produces a ranking reference report, not a probabilistic classifier bundle.
- `train`/`tune` use development folds only. `calibrate` uses the calibration partition and checks mappings on policy rows.
- Freeze writes the policy and final bundle manifest, then hashes them into the freeze manifest.
- Every modelling command logs local MLflow runs and a machine-readable run manifest.

T06 preserves the fixed unweighted C=1 LR reference at `artifacts/baselines/logistic_reference/`. Calibration and policy freeze also write its independent mapping/cutoff and hashes, so final policy comparisons have consistent capacity semantics.

## 5. Final evaluation and explanation

```powershell
& $fgPython -m fraudguard.cli evaluate --artifact-dir artifacts/champion --baseline-dir artifacts/baselines --split-manifest data/processed/split_manifest.json --partition final_test --freeze-manifest reports/freeze_manifest.json --output-dir reports/final
& $fgPython -m fraudguard.cli explain --artifact-dir artifacts/champion --split-manifest data/processed/split_manifest.json --partition policy --max-rows 200 --output-dir reports/explanations
```

`evaluate` must refuse a missing/mismatched freeze manifest. A rerun requires an explicit labelled replay mode `--replay`; it records that the holdout has already been viewed. It cannot overwrite prior evidence silently or claim a new unbiased evaluation.

Evaluation can predict large partitions in chunks but computes ranking/cap metrics across the complete partition. CLI batch-scoring limits do not force a truncated final evaluation.

## 6. Score and serve

```powershell
& $fgPython -m fraudguard.cli score --input examples/synthetic_batch.csv --artifact-dir artifacts/champion --review-fraction 0.01 --output-dir reports/demo
$env:FRAUDGUARD_ARTIFACT_DIR = (Resolve-Path 'artifacts/champion').Path
& $fgPython -m uvicorn fraudguard.api:app --host 127.0.0.1 --port 8000
```

In another PowerShell terminal:

```powershell
Invoke-RestMethod -Uri 'http://127.0.0.1:8000/health'
Invoke-RestMethod -Uri 'http://127.0.0.1:8000/ready'
Invoke-RestMethod -Uri 'http://127.0.0.1:8000/model-info'
$fgPayload = Get-Content 'examples/synthetic_transaction.json' -Raw
Invoke-RestMethod -Method Post -Uri 'http://127.0.0.1:8000/v1/score' -ContentType 'application/json' -Body $fgPayload
```

T08 supplies the synthetic JSON example. T15 supplies the synthetic CSV. Example output scores are measured from the currently loaded local model; examples do not demonstrate real transaction fraud.

## 7. Monitoring and benchmark

```powershell
& $fgPython -m fraudguard.cli monitor --input examples/synthetic_batch.csv --artifact-dir artifacts/champion --output-dir reports/monitoring
& $fgPython -m fraudguard.cli benchmark --url http://127.0.0.1:8000/v1/score --input examples/synthetic_transaction.json --warmups 100 --requests 1000 --concurrency 1 --output reports/api_benchmark.json
& $fgPython -m fraudguard.cli reproduce --artifact-dir artifacts/champion --freeze-manifest reports/freeze_manifest.json --split-manifest data/processed/split_manifest.json --output-dir reports/repeatability
& $fgPython -m mlflow server --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
```

Use the same local SQLite tracking URI in config and the MLflow viewing server. Verify the chosen installed MLflow version's behaviour at T01. No public tracking server is required.

`reproduce` retrains the frozen configuration on development rows, refits the previously selected mapping on calibration rows if applicable, and compares development/policy metrics. It preserves the recorded mapping choice and cutoff, does not tune or read final-test labels, and never overwrites the original champion bundle.

## 8. Quality checks

```powershell
& $fgPython -m ruff check src tests
& $fgPython -m ruff format --check src tests
& $fgPython -m pytest -p no:cacheprovider --basetemp .tmp/pytest --cov=fraudguard --cov-report=term-missing
```

Keep test temporary files inside the project. Targeted per-task tests are listed in the task tracker. Expand tests when new failures or risks justify it.

## 9. Local container check

```powershell
docker build -t fraudguard:local .
$fgArtifacts = (Resolve-Path 'artifacts/champion').Path
docker run --rm -p 127.0.0.1:8000:8000 --mount "type=bind,source=$fgArtifacts,target=/app/artifacts/champion,readonly" -e FRAUDGUARD_ARTIFACT_DIR=/app/artifacts/champion fraudguard:local
```

The container entrypoint binds to `0.0.0.0` internally. The host port is restricted to localhost. Verify readiness and a synthetic score, then stop the foreground process when the check finishes.

## 10. Failure recovery

1. Record the exact command, error category and affected task.
2. Do not delete original data, completed reports or unrelated changes.
3. Correct the responsible contract/config/code and rerun the affected check.
4. If access, consent or Docker is unavailable, leave the gate open and continue independent ready work.
5. Never treat credential printing, global installation or repeated test retuning as a workaround.
