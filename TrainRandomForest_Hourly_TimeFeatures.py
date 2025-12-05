import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# LOAD DATA
df = pd.read_csv("final_data_hourly_timeFeatures.csv", index_col="datetime", parse_dates=True)

print("Loaded dataset shape:", df.shape)
print("Date range:", df.index.min(), "→", df.index.max())

# FEATURE SETUP
target = "Global_active_power"

# All features EXCEPT the target
feature_cols = [col for col in df.columns if col != target]

X = df[feature_cols]
y = df[target]

# TIME-AWARE SPLIT (Train old data → Test new data)
test_start_date = "2010-02-05 00:00:00"

X_train = X[X.index < test_start_date]
X_test  = X[X.index >= test_start_date]
y_train = y[y.index < test_start_date]
y_test  = y[y.index >= test_start_date]

print("\nTrain size:", len(X_train), " Test size:", len(X_test))
print("Train range:", X_train.index.min(), "→", X_train.index.max())
print("Test range: ", X_test.index.min(),  "→", X_test.index.max())

# TRAIN RANDOM FOREST
rf = RandomForestRegressor(
    n_estimators=300,
    max_depth=20,
    min_samples_split=5,
    n_jobs=-1,
    random_state=42
)

rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

# METRICS
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\nRandom Forest (Hourly + Time Features) Results")
print(f"MAE:  {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R²:   {r2:.4f}")


# PLOTS
# Actual vs Predicted 
plt.figure(figsize=(8, 8))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.3)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
         'r--', label="Perfect Fit Line")
plt.xlabel("Actual Global Active Power")
plt.ylabel("Predicted Global Active Power")
plt.title("Random Forest (Hourly + Time Features): Actual vs Predicted")
plt.grid(True, linestyle="--", alpha=0.4)
plt.legend()
plt.tight_layout()
plt.show()

# Residuals 
residuals = y_test - y_pred

plt.figure(figsize=(10, 5))
sns.scatterplot(x=y_pred, y=residuals, alpha=0.3)
plt.axhline(0, linestyle='--', color='red')
plt.xlabel("Predicted Values")
plt.ylabel("Residuals (Actual - Predicted)")
plt.title("Random Forest (Hourly + Time Features): Residuals vs Predictions")
plt.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.show()

# Residual Distribution 
plt.figure(figsize=(10, 5))
sns.histplot(residuals, bins=50, kde=True)
plt.title("Random Forest (Hourly + Time Features): Residual Distribution")
plt.xlabel("Residual")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()


# Feature Importance 
importances = pd.Series(rf.feature_importances_, index=feature_cols).sort_values()

plt.figure(figsize=(12, 8))
sns.barplot(x=importances.values, y=importances.index, palette="Blues", linewidth=1, edgecolor="black")
plt.title("Random Forest Feature Importance (Hourly + Time Features)")
plt.xlabel("Feature Importance")
plt.ylabel("Feature")

# Add labels
for i, v in enumerate(importances.values):
    plt.text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=10)

plt.tight_layout()
plt.show()
