# AI use log (notes for the README's "AI use" section)

## Written by AI (Claude Code)
- Project setup: `uv init --python 3.11`, `uv add pycaret six "fastapi[standard]"` (with `UV_HTTP_TIMEOUT=300`)
- Downloaded the Telco CSV into `data/` from IBM's GitHub mirror (the same file as the Kaggle BlastChar dataset)
- All of `discovery.py`:
  - data cleaning (drop `customerID`, convert `TotalCharges` to a number)
  - the PyCaret section, including rendering the comparison table to a PNG and moving the confusion matrix into `images/`
  - the scikit-learn pipeline
- Checked the rebuild against PyCaret's best model: `LogisticRegression(max_iter=1000, random_state=42)`, other parameters left at their defaults

## Written by me
- (fill in: e.g. main.py, README answers, prediction experiment, ...)

## Things to check or fix in the AI's work
- PyCaret's Recall / Prec. / F1 columns are **weighted averages over both classes**, which is why Recall equals Accuracy. They are not recall for the churn class. The churn-class ("Yes") recall from scikit-learn is 0.5615.
- In the confusion matrix, label 1 = Churn "Yes" (PyCaret encodes the target as 0/1).
- PyCaret run time changes between runs (433 s, then 563 s), and LightGBM takes most of it.
