import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load cleaned dataset
df = pd.read_csv("final_data.csv", index_col="datetime", parse_dates=True)
df = df.dropna()

# Sort chronologically to avoid data leakage
df = df.sort_index()

# Define features and target
X = df[["Voltage", "Global_reactive_power",
        "Sub_metering_1", "Sub_metering_2", "Sub_metering_3"]]
y = df["Global_active_power"]

# Time-based split (80% train, 20% test)
split_idx = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

print(f"Train range: {X_train.index.min()} → {X_train.index.max()}")
print(f"Test range:  {X_test.index.min()} → {X_test.index.max()}")
print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

# Random Forest with tuned parameters
rf = RandomForestRegressor(
    n_estimators=400,
    max_depth=16,
    min_samples_leaf=2,
    max_features="sqrt",
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\nRandom Forest (Time-Aware Split) Results")
print(f"MAE:  {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R²:   {r2:.4f}")


# Actual vs Predicted Plot (Time-Aware)
plt.figure(figsize=(8, 8))
sns.set(style="white")

# sample 5,000 points for readability
sample_idx = np.random.choice(len(y_test), size=5000, replace=False)

y_test_sample = y_test.iloc[sample_idx]
y_pred_sample = y_pred[sample_idx]

plt.scatter(
    y_test_sample,
    y_pred_sample,
    alpha=0.3,
    color="#1f77b4",
    edgecolor="none"
)

# Perfect fit line
low = min(y_test_sample.min(), y_pred_sample.min())
high = max(y_test_sample.max(), y_pred_sample.max())
plt.plot([low, high], [low, high], "r--", label="Perfect Fit Line")

plt.title("Random Forest: Actual vs Predicted (Time-Aware Split)",
          fontsize=18, weight="bold")
plt.xlabel("Actual Global Active Power")
plt.ylabel("Predicted Global Active Power")
plt.grid(True, linestyle="--", alpha=0.4)
plt.legend()
plt.tight_layout()
plt.show()


# Residuals vs Predictions
residuals = y_test.values - y_pred

plt.figure(figsize=(10, 5))
sns.scatterplot(
    x=y_pred[:20000],     # limit for readability
    y=residuals[:20000],
    alpha=0.25,
    color="#1f77b4"
)

plt.axhline(0, color="red", linestyle="--")
plt.title("Random Forest Residuals vs Predictions (Time-Aware Split)",
          fontsize=18, weight="bold")
plt.xlabel("Predicted Values")
plt.ylabel("Residuals (Actual - Predicted)")
plt.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.show()


# Residual Distribution
plt.figure(figsize=(10, 5))
sns.histplot(residuals, bins=50, kde=True, color="#1f77b4")

plt.title("Random Forest Residual Distribution", fontsize=18, weight="bold")
plt.xlabel("Residual")
plt.ylabel("Frequency")
plt.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.show()


# Feature Importance (sorted, professional style)
importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values()

plt.figure(figsize=(12, 7))
sns.set(style="white")

ax = sns.barplot(
    x=importances.values,
    y=importances.index,
    palette="Blues",
    linewidth=1,
    edgecolor="black"
)

# value labels
for i, v in enumerate(importances.values):
    ax.text(v + 0.01, i, f"{v:.3f}",
            fontsize=11, va="center",
            color="black", fontweight="bold")

plt.title("Random Forest Feature Importance (Tuned, Time-Aware)",
          fontsize=20, weight="bold", pad=20)
plt.xlabel("Feature Importance")
plt.ylabel("Feature")

ax.xaxis.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.show()

