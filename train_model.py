"""
train_model.py — trains XGBoost on hotel booking data and saves model artifacts
Run: python train_model.py
Outputs: model.pkl, feature_columns.pkl, label_encoders.pkl
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, roc_auc_score
)
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings("ignore")

# ── Load ───────────────────────────────────────────────────────────────────────
df = pd.read_csv("hotel_bookings.csv")
print(f"Loaded {df.shape[0]:,} rows × {df.shape[1]} columns")

# ── Drop leaky and useless columns ────────────────────────────────────────────
DROP = [
    "reservation_status",      # directly leaks target
    "reservation_status_date", # leaks target
    "agent",                   # 94% missing
    "company",                 # 94% missing
    "arrival_date_year",       # not predictive for future
    "arrival_date_week_number",
    "arrival_date_day_of_month",
]
df.drop(columns=[c for c in DROP if c in df.columns], inplace=True)

# ── Remove impossible rows (no guests) ────────────────────────────────────────
df = df[~((df["adults"] == 0) & (df["children"] == 0) & (df["babies"] == 0))]

# ── Fill nulls ─────────────────────────────────────────────────────────────────
df["children"].fillna(0, inplace=True)
df["country"].fillna(df["country"].mode()[0], inplace=True)
df.fillna(0, inplace=True)

# ── Feature engineering ───────────────────────────────────────────────────────
df["total_nights"]   = df["stays_in_week_nights"] + df["stays_in_weekend_nights"]
df["total_guests"]   = df["adults"] + df["children"] + df["babies"]
df["is_family"]      = ((df["adults"] > 0) & (df["children"] > 0)).astype(int)
df["room_changed"]   = (df["reserved_room_type"] != df["assigned_room_type"]).astype(int)
df["is_weekend_only"]= ((df["stays_in_weekend_nights"] > 0) & (df["stays_in_week_nights"] == 0)).astype(int)
df["revenue"]        = df["adr"] * df["total_nights"]
df["lead_time_log"]  = np.log1p(df["lead_time"])
df["adr_log"]        = np.log1p(df["adr"])

# Month to ordinal
month_order = ["January","February","March","April","May","June",
               "July","August","September","October","November","December"]
df["arrival_month_num"] = df["arrival_date_month"].map(
    {m: i+1 for i, m in enumerate(month_order)}
)

# Drop raw columns replaced by engineered ones
df.drop(columns=["adults","children","babies","stays_in_week_nights",
                  "stays_in_weekend_nights","lead_time","adr",
                  "arrival_date_month"], inplace=True)

# ── Encode categoricals ────────────────────────────────────────────────────────
CAT_COLS = [c for c in df.columns if df[c].dtype == "object" and c != "is_canceled"]
label_encoders = {}
for col in CAT_COLS:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

print(f"Features after engineering: {df.shape[1]-1}")

# ── Split ──────────────────────────────────────────────────────────────────────
X = df.drop("is_canceled", axis=1)
y = df["is_canceled"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {X_train.shape[0]:,}  Test: {X_test.shape[0]:,}")
print(f"Cancellation rate: {y.mean():.1%}")

# ── Train XGBoost ─────────────────────────────────────────────────────────────
print("\nTraining XGBoost...")
model = XGBClassifier(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=3,
    gamma=0.1,
    reg_alpha=0.1,
    reg_lambda=1.0,
    scale_pos_weight=1,
    use_label_encoder=False,
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1,
)
model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=100,
)

# ── Evaluate ──────────────────────────────────────────────────────────────────
y_pred  = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

acc     = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)
cm      = confusion_matrix(y_test, y_pred)

print(f"\n{'='*50}")
print(f"  RESULTS")
print(f"{'='*50}")
print(f"  Accuracy : {acc:.4f} ({acc*100:.2f}%)")
print(f"  ROC-AUC  : {roc_auc:.4f}")
print(f"\nConfusion Matrix:\n{cm}")
print(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")

# CV score
cv = cross_val_score(model, X, y, cv=5, scoring="accuracy", n_jobs=-1)
print(f"\n5-Fold CV Accuracy: {cv.mean():.4f} ± {cv.std():.4f}")

# Feature importance top 15
fi = pd.Series(model.feature_importances_, index=X.columns)
print(f"\nTop 15 Features:\n{fi.nlargest(15).to_string()}")

# ── Save artifacts ────────────────────────────────────────────────────────────
joblib.dump(model,          "model.pkl")
joblib.dump(list(X.columns),"feature_columns.pkl")
joblib.dump(label_encoders, "label_encoders.pkl")

print(f"\nSaved: model.pkl  feature_columns.pkl  label_encoders.pkl")
