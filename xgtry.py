import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
from xgboost import XGBClassifier

TRAIN_PATH = "train.csv"
TEST_PATH  = "test.csv"
OUT_PATH   = "submission.csv"

TARGET   = "urgency_level"
ID_COL   = "id"
TIME_COL = "step"
CAT_COLS = ["type"]

# Load files
train = pd.read_csv(TRAIN_PATH)
test  = pd.read_csv(TEST_PATH)

# strip any accidental whitespace in column names
train.columns = train.columns.str.strip()
test.columns  = test.columns.str.strip()

# Light cleanup for columns
for c in CAT_COLS:
    if c in train.columns:
        train[c] = train[c].astype(str)
        test[c]  = test[c].astype(str)

# Drop raw IDs (high-cardinality strings)
DROP_COLS = ["nameOrig", "nameDest"]
train = train.drop(columns=[c for c in DROP_COLS if c in train.columns])
test  = test.drop(columns=[c for c in DROP_COLS if c in test.columns])

# Time-based split (last 20% of steps as validation)
cutoff = np.quantile(train[TIME_COL], 0.80)
tr_idx = train[TIME_COL] <= cutoff
va_idx = train[TIME_COL] > cutoff

train_tr = train.loc[tr_idx].copy()
train_va = train.loc[va_idx].copy()

# Separate labels
y_tr = train_tr[TARGET].astype(int)
y_va = train_va[TARGET].astype(int)

# Drop target/id from features
X_tr = train_tr.drop(columns=[TARGET, ID_COL])
X_va = train_va.drop(columns=[TARGET, ID_COL])
X_test = test.drop(columns=[ID_COL])

# One-hot encode categoricals (type) and align columns across splits
X_tr = pd.get_dummies(X_tr, columns=CAT_COLS, dummy_na=True)
X_va = pd.get_dummies(X_va, columns=CAT_COLS, dummy_na=True)
X_test = pd.get_dummies(X_test, columns=CAT_COLS, dummy_na=True)

# Align so all have identical columns
X_tr, X_va = X_tr.align(X_va, join="left", axis=1, fill_value=0)
X_tr, X_test = X_tr.align(X_test, join="left", axis=1, fill_value=0)

# Per-row sample weights (better than class_weights list for XGBoost)
counts = y_tr.value_counts().sort_index()
total = counts.sum()
class_w = (total / (len(counts) * counts)).to_dict()
w_tr = y_tr.map(class_w).astype(float).values

# Train XGBoost multiclass
model = XGBClassifier(
    objective="multi:softprob",
    num_class=4,
    n_estimators=2500,
    learning_rate=0.05,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=5.0,
    min_child_weight=10,
    tree_method="hist",   
    random_state=42,
    eval_metric="mlogloss")

model.fit(
    X_tr, y_tr,
    sample_weight=w_tr,
    eval_set=[(X_va, y_va)],
    verbose=200)

# Validate 
va_probs = model.predict_proba(X_va)
va_pred = va_probs.argmax(axis=1)
print("Validation macro-F1:", f1_score(y_va, va_pred, average="macro"))

# Prediction
test_probs = model.predict_proba(X_test)
test_pred = test_probs.argmax(axis=1)

submission = pd.DataFrame({ID_COL: test[ID_COL], TARGET: test_pred})
submission.to_csv(OUT_PATH, index=False)
print(f"Wrote {OUT_PATH}")
