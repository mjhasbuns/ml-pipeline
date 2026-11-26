"""Compare a low-code (PyCaret) and a manual (scikit-learn) workflow on Telco Customer Churn."""

import shutil
import time
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")  # write PNGs, never open windows
import matplotlib.pyplot as plt
import pandas as pd

RAW = Path("data/Telco-Customer-Churn.csv")
CLEAN = Path("data/telco_churn_clean.csv")
IMAGES = Path("images")
IMAGES.mkdir(exist_ok=True)
TARGET = "Churn"

# ---------------------------------------------------------------- data prep
df = pd.read_csv(RAW)
df = df.drop(columns=["customerID"])  # ID column: no predictive meaning
# TotalCharges is text with 11 blanks (brand-new customers, tenure 0) -> NaN, imputed later
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df.to_csv(CLEAN, index=False)
print(f"Data: {df.shape[0]} rows, {df.shape[1] - 1} features")
print(df[TARGET].value_counts(normalize=True).round(3), "\n")


# ---------------------------------------------------------------- PyCaret
def save_table_png(table: pd.DataFrame, path: Path) -> None:
    """Render the compare_models() table as an image."""
    fig, ax = plt.subplots(figsize=(14, 0.45 * len(table) + 1))
    ax.axis("off")
    t = ax.table(cellText=table.values, colLabels=table.columns, loc="center", cellLoc="center")
    t.auto_set_font_size(False)
    t.set_fontsize(9)
    t.scale(1, 1.4)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


from pycaret.classification import compare_models, plot_model, pull, setup

start = time.perf_counter()

setup(data=df, target=TARGET, session_id=42, verbose=False)
top3 = compare_models(n_select=3, verbose=False)
comparison = pull()
best = top3[0]

plot_model(best, plot="confusion_matrix", save=True)  # writes "Confusion Matrix.png"

pycaret_seconds = time.perf_counter() - start

print("PyCaret comparison (10-fold CV on the 70% training split):")
print(comparison.to_string(), "\n")
comparison.to_csv(IMAGES / "pycaret_comparison.csv")
save_table_png(comparison.reset_index(drop=True).round(4), IMAGES / "pycaret_comparison.png")
shutil.move("Confusion Matrix.png", IMAGES / "confusion_matrix.png")
print("Top 3:", [type(m).__name__ for m in top3])
print("Best model:", best)
print(f"PyCaret section took {pycaret_seconds:.1f} s\n")

# ---------------------------------------------------------------- scikit-learn
# Rebuild PyCaret's best model (Logistic Regression) by hand.
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

start = time.perf_counter()

X = df.drop(columns=[TARGET])
y = df[TARGET]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

numeric = X.select_dtypes(include="number").columns.tolist()
categorical = X.select_dtypes(exclude="number").columns.tolist()

preprocess = ColumnTransformer([
    ("num", Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ]), numeric),
    ("cat", Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]), categorical),
])

model = Pipeline([
    ("preprocess", preprocess),
    ("clf", LogisticRegression(max_iter=1000, random_state=42)),  # same settings as PyCaret's lr
])
model.fit(X_train, y_train)

print("scikit-learn Logistic Regression on the 30% held-out test set:")
print(classification_report(y_test, model.predict(X_test), digits=4))

joblib.dump(model, "model.joblib")
sklearn_seconds = time.perf_counter() - start
print(f"scikit-learn section took {sklearn_seconds:.1f} s")
print(f"Numeric features: {numeric}")
print(f"Categorical features: {categorical}")
