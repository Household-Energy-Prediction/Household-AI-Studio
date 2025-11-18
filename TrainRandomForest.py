import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# load cleaned dataset
df = pd.read_csv("final_data.csv", index_col = "datetime", parse_dates = True)

# drop missing values for consistency
df = df.dropna()

# define features and target 
X = df[["Voltage", "Global_reactive_power",
        "Sub_metering_1", "Sub_metering_2", "Sub_metering_3"]]

y = df["Global_active_power"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size = 0.2, random_state = 42
)

# fine-tuned Random Forest Model
rf = RandomForestRegressor(
    n_estimators = 400,      # more trees = more stable
    max_depth = 16,          # deeper trees
    min_samples_leaf = 2,    # prevents overfitting
    max_features = "sqrt",   # improves generalization
    random_state = 42,
    n_jobs = -1
)

rf.fit(X_train, y_train)

# predictions & evaluation
y_pred = rf.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\nFine-Tuned Random Forest Model Results")
print(f"MAE:  {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R²:   {r2:.4f}")


# Actual vs Predicted Plot
plt.figure(figsize=(8, 8))
sns.set(style="white")

sample_idx = np.random.choice(len(y_test), size=5000, replace=False)

plt.scatter(
    y_test.iloc[sample_idx],
    y_pred[sample_idx],
    alpha=0.3,
    edgecolor='none',
    color="#1f77b4"
)

min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--')

plt.title("Random Forest: Actual vs Predicted", fontsize=18, weight="bold")
plt.xlabel("Actual Global Active Power")
plt.ylabel("Predicted Global Active Power")
plt.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()
plt.show()

# Residual Plot
residuals = y_test - y_pred

plt.figure(figsize=(10, 5))
sns.scatterplot(
    x=y_pred,
    y=residuals,
    alpha=0.2,
    color="#1f77b4"
)

plt.axhline(0, color="red", linestyle="--")
plt.title("Random Forest Residuals vs Predictions", fontsize=18, weight="bold")
plt.xlabel("Predicted Values")
plt.ylabel("Residuals (Actual - Predicted)")
plt.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()
plt.show()

#  Residual Histogram
plt.figure(figsize=(8, 5))
sns.histplot(residuals, bins=50, kde=True, color="#1f77b4")

plt.title("Random Forest Residual Distribution", fontsize=18, weight="bold")
plt.xlabel("Residual")
plt.ylabel("Frequency")
plt.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()
plt.show()



# Sort features from least → most important
importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values()

plt.figure(figsize=(12, 7))
sns.set(style="white")

# Barplot
ax = sns.barplot(
    x=importances.values,
    y=importances.index,
    palette="Blues",     # darkest = most important
    linewidth=1,
    edgecolor="black"
)

# Value labels
for i, v in enumerate(importances.values):
    ax.text(v + 0.01, i, f"{v:.3f}",
            color="black", va="center",
            fontsize=11, fontweight="bold")

# Titles & labels
plt.title("Random Forest Feature Importance (Tuned)", fontsize=20, weight="bold", pad=20)
plt.xlabel("Feature Importance", fontsize=14)
plt.ylabel("Feature", fontsize=14)

# Cleaner grid
ax.xaxis.grid(True, linestyle="--", alpha=0.4)
ax.yaxis.grid(False)

plt.tight_layout()
plt.show()
