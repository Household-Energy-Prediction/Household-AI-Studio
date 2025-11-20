import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# -----------------------------
# 1. LOAD HOURLY DATA
# -----------------------------
df = pd.read_csv("final_data_hourly.csv", index_col="datetime", parse_dates=True)
df = df.dropna()

print("Hourly data shape:", df.shape)
print("Hourly date range:", df.index.min(), "→", df.index.max())

# -----------------------------
# 2. FEATURES / TARGET
# -----------------------------
X = df[["Voltage",
        "Global_reactive_power",
        "Sub_metering_1",
        "Sub_metering_2",
        "Sub_metering_3"]]

y = df["Global_active_power"]

# -----------------------------
# 3. TIME-AWARE SPLIT (80/20)
# -----------------------------
split_index = int(len(df) * 0.8)

X_train = X.iloc[:split_index]
X_test  = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test  = y.iloc[split_index:]

print("\nTrain range:", X_train.index.min(), "→", X_train.index.max())
print("Test range: ", X_test.index.min(),  "→", X_test.index.max())
print("Train size:", len(X_train), "Test size:", len(X_test))

# -----------------------------
# 4. RANDOM FOREST MODEL
# -----------------------------
rf = RandomForestRegressor(
    n_estimators=400,
    max_depth=16,
    min_samples_leaf=2,
    max_features="sqrt",
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)

# -----------------------------
# 5. PREDICTIONS + METRICS
# -----------------------------
y_pred = rf.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\nRandom Forest (Hourly, Time-Aware Split) Results")
print(f"MAE:  {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R²:   {r2:.4f}")

# -----------------------------
# 6. PLOTS
# -----------------------------
sns.set(style="white")

# ----- A. Actual vs Predicted -----
plt.figure(figsize=(8, 8))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.4)

min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', label="Perfect Fit Line")

plt.title("Random Forest (Hourly): Actual vs Predicted", fontsize=18, weight="bold")
plt.xlabel("Actual Global Active Power")
plt.ylabel("Predicted Global Active Power")
plt.grid(True, linestyle='--', alpha=0.4)
plt.legend()
plt.tight_layout()
plt.show()

# ----- B. Residuals vs Predictions -----
residuals = y_test - y_pred

plt.figure(figsize=(10, 5))
sns.scatterplot(x=y_pred, y=residuals, alpha=0.4)
plt.axhline(0, color='red', linestyle='--')

plt.title("Random Forest (Hourly): Residuals vs Predictions", fontsize=18, weight="bold")
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()
plt.show()

# ----- C. Residual Distribution -----
plt.figure(figsize=(10, 5))
sns.histplot(residuals, bins=40, kde=True, color="#1f77b4")

plt.title("Random Forest (Hourly): Residual Distribution", fontsize=18, weight="bold")
plt.xlabel("Residual")
plt.ylabel("Frequency")
plt.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()
plt.show()

# ----- D. Feature Importance -----
importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values()

plt.figure(figsize=(12, 7))
ax = sns.barplot(x=importances.values, y=importances.index,
                 palette="Blues", linewidth=1, edgecolor="black")

for i, v in enumerate(importances.values):
    ax.text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=11, fontweight="bold")

plt.title("Random Forest Feature Importance (Hourly, Time-Aware)", fontsize=20, weight="bold")
plt.xlabel("Feature Importance")
plt.ylabel("Feature")
ax.xaxis.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.show()
