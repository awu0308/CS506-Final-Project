"""
NBA Career Longevity Predictor - Modeling
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import classification_report, confusion_matrix
import os

os.makedirs("plots", exist_ok=True)

# Load and prepare data
df = pd.read_csv("nba_career_longevity_data.csv")

# Features to use
_ALL_FEATURES = [
    'YEARS_IN_LEAGUE',
    'GP',             
    'MPG',
    'PPG',
    'RPG',
    'APG',
    'PIE',          
    'CUMULATIVE_MINUTES',
    'CUMULATIVE_GAMES',
    'CAREER_PEAK_PPG',
    'DISTANCE_FROM_PEAK',
    'PPG_CHANGE',
    'MPG_CHANGE',
]
FEATURES = [f for f in _ALL_FEATURES if f in df.columns]

TARGET = 'REMAINING_COMPETITIVE_SEASONS'

# Filter to competitive seasons only 
df_model = df[df['IS_COMPETITIVE']].copy()

# Drop rows with NaN in features (first season won't have PPG_CHANGE, etc.)
df_model = df_model.dropna(subset=FEATURES + [TARGET])

print(f"Modeling dataset: {len(df_model)} records, {df_model['PLAYER_ID'].nunique()} players")
print(f"Features: {FEATURES}")

X = df_model[FEATURES]
y = df_model[TARGET]

# Regression (predict exact remaining years)
print("\n" + "=" * 50)
print("APPROACH 1: REGRESSION")
print("=" * 50)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

rf_reg = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
rf_reg.fit(X_train, y_train)
y_pred = rf_reg.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"  Mean Absolute Error: {mae:.2f} seasons")
print(f"  RMSE: {rmse:.2f} seasons")
print(f"  R² Score: {r2:.3f}")

# Cross-validation
cv_scores = cross_val_score(rf_reg, X, y, cv=5, scoring='neg_mean_absolute_error')
print(f"  5-Fold CV MAE: {-cv_scores.mean():.2f} ± {cv_scores.std():.2f}")

# Baseline: Linear Regression
print("\n--- Baseline: Linear Regression ---")
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
lr_mae = mean_absolute_error(y_test, y_pred_lr)
lr_rmse = np.sqrt(mean_squared_error(y_test, y_pred_lr))
lr_r2 = r2_score(y_test, y_pred_lr)
print(f"  MAE:  {lr_mae:.2f} seasons")
print(f"  RMSE: {lr_rmse:.2f} seasons")
print(f"  R²:   {lr_r2:.3f}")
print(f"\n  Random Forest improvement over Linear Regression:")
print(f"  MAE  {lr_mae:.2f} → {mae:.2f} ({((lr_mae - mae) / lr_mae * 100):.1f}% better)")

# Predicted vs Actual
fig, ax = plt.subplots()
ax.scatter(y_test, y_pred, alpha=0.3, s=10, color='steelblue')
ax.plot([0, y_test.max()], [0, y_test.max()], 'r--', label='Perfect prediction')
ax.set_xlabel("Actual Remaining Competitive Seasons")
ax.set_ylabel("Predicted Remaining Competitive Seasons")
ax.set_title(f"Predicted vs Actual (MAE={mae:.2f}, R²={r2:.3f})")
ax.legend()
plt.tight_layout()
plt.savefig("plots/07_predicted_vs_actual_regression.png")
plt.close()
print("  Saved: 07_predicted_vs_actual_regression.png")

# Feature Importance
importances = pd.Series(rf_reg.feature_importances_, index=FEATURES).sort_values()
fig, ax = plt.subplots(figsize=(8, 6))
importances.plot(kind='barh', color='steelblue', edgecolor='black', ax=ax)
ax.set_xlabel("Feature Importance")
ax.set_title("Random Forest Feature Importance (Regression)")
plt.tight_layout()
plt.savefig("plots/08_feature_importance_regression.png")
plt.close()
print("  Saved: 08_feature_importance_regression.png")

# Residual distribution 
residuals = y_test - y_pred
fig, ax = plt.subplots()
ax.hist(residuals, bins=30, edgecolor='black', color='steelblue')
ax.axvline(x=0, color='red', linestyle='--')
ax.set_xlabel("Residual (Actual - Predicted)")
ax.set_ylabel("Count")
ax.set_title("Residual Distribution")
plt.tight_layout()
plt.savefig("plots/09_residual_distribution.png")
plt.close()
print("  Saved: 09_residual_distribution.png")

# Classification (bucket into categories)
print("\n" + "=" * 50)
print("APPROACH 2: CLASSIFICATION")
print("=" * 50)

# buckets
def bucket_years(y_val):
    if y_val <= 1:
        return "0-1 years"
    elif y_val <= 3:
        return "2-3 years"
    elif y_val <= 5:
        return "4-5 years"
    else:
        return "6+ years"

y_bucketed = y.apply(bucket_years)
bucket_order = ["0-1 years", "2-3 years", "4-5 years", "6+ years"]

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X, y_bucketed, test_size=0.2, random_state=42
)

rf_clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
rf_clf.fit(X_train_c, y_train_c)
y_pred_c = rf_clf.predict(X_test_c)

print("\nClassification Report:")
print(classification_report(y_test_c, y_pred_c, labels=bucket_order))

# Confusion Matrix
cm = confusion_matrix(y_test_c, y_pred_c, labels=bucket_order)
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=bucket_order, yticklabels=bucket_order, ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title("Confusion Matrix - Career Longevity Buckets")
plt.tight_layout()
plt.savefig("plots/10_confusion_matrix.png")
plt.close()
print("  Saved: 10_confusion_matrix.png")

# Summary 
print("\n" + "=" * 50)
print("=" * 50)
print
(f"""
Dataset:
  {len(df_model)} competitive player-seasons from {df_model['PLAYER_ID'].nunique()} players

Regression (Random Forest):
  MAE: {mae:.2f} seasons
  R²:  {r2:.3f}
  The model predicts remaining competitive years within ~{mae:.1f} seasons on average.

Classification (Random Forest):
  See classification report above for per-bucket precision/recall.

Top 3 Most Important Features:
  {importances.index[-1]}: {importances.values[-1]:.3f}
  {importances.index[-2]}: {importances.values[-2]:.3f}
  {importances.index[-3]}: {importances.values[-3]:.3f}
""")
