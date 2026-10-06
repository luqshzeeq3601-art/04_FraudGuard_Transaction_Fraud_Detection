# FraudGuard working instructions

## 1. Communication

- Explain directly and briefly, using numbered steps and useful bullets.
- State the outcome, evidence, limitation and next ready task.

## 2. Before every execution session

1. Read [START_HERE](docs/00_START_HERE.md), [progress](docs/09_PROGRESS_LOG.md), [decisions](docs/08_DECISIONS_LOG.md) and [tasks](tasks/todo.md).
2. Read the PRD and the owning specifications for the selected task.
3. Inspect the current files and Git state. Preserve unrelated changes and neighbouring projects.
4. Execute the first unfinished task whose dependencies are complete. A user instruction to start or continue authorizes the documented local implementation; do not repeatedly seek approval for routine steps within scope.

## 3. Engineering rules

- Markdown is the implementation contract. Do not require this chat or the original spreadsheet to resume.
- Requirements belong in the PRD, data rules in the data spec, model selection in the experiment plan, and task completion only in `tasks/todo.md`.
- Resolve document conflicts before the affected implementation. Record material changes and update all affected contracts. User instructions take precedence.
- Keep reusable code in `src/fraudguard/`; notebooks must call package logic.
- Use type hints, snake_case functions, small modules and meaningful pytest tests.
- Verify installed-version APIs against official documentation and pin dependencies at T01.
- Use an isolated environment. Do not install globally or modify Projects 1 and 2.
- Fit preprocessing only on the permitted training partition. Never shuffle the primary evaluation split or use labels, transaction identifiers or future aggregates as predictors.
- Final-test labels are for T13 evaluation only. Do not tune against them or claim repeated test use as a fresh holdout.
- Model bundles are locally generated trusted files. Never load uploaded pickle/joblib/model files.
- Schema errors must be visible. Do not silently coerce malformed requests or default failed scores to zero.
- Record missed objectives honestly. Do not lower targets after seeing final-test results.

## 4. Scope and authorization

- This initial session creates planning files only.
- After a start instruction, proceed with local setup, authorized dataset acquisition, implementation, experiments, tests and reports described here.
- Ask for direction before changing the dataset, label meaning, primary split, review policy or MVP scope.
- If Kaggle requires account consent, credentials or terms acceptance, let the user perform that step. Never print credentials or bypass access controls.
- Public publication, paid services, external messages and live payment integration require authorization unless already explicitly authorized.
- Dashboards, streaming infrastructure, automated blocking and automatic retraining are outside the MVP.

## 5. End every execution session

1. Complete task checkboxes only when acceptance checks have evidence.
2. Append commands, results, changed files and unresolved issues to the progress log.
3. Record material decisions with alternatives and implications in the decision log.
4. Leave the next task ID and its prerequisites. Do not claim readiness from an unimplemented command or a document alone.
