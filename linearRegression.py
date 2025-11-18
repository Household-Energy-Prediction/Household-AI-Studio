import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load final dataset
df = pd.read_csv("final_data.csv", index_col="datetime", parse_dates=True)
df = df.dropna()
df = df.sort_index()  # critical for forecasting

# Features & target
X = df[["Voltage", "Global_reactive_power",
        "Sub_metering_1", "Sub_metering_2", "Sub_metering_3"]]
y = df["Global_active_power"]

# Time split (not random)
split_idx = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

# Scale inputs
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train LR
lin_reg = LinearRegression()
lin_reg.fit(X_train_scaled, y_train)

# Predict
y_pred = lin_reg.predict(X_test_scaled)

# Evaluation
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\nLinear Regression (Time-Aware Split) Results")
print(f"MAE:  {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R²:   {r2:.4f}")


# Coefficient Plot (Feature Influence)
coef_df = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": lin_reg.coef_
})

# Sort by absolute magnitude of coefficient
coef_df["AbsCoef"] = coef_df["Coefficient"].abs()
coef_df = coef_df.sort_values(by="AbsCoef", ascending=True)

plt.figure(figsize=(12, 7))
sns.set(style="white")

ax = sns.barplot(
    x="Coefficient",
    y="Feature",
    data=coef_df,
    palette="Blues",
    linewidth=1,
    edgecolor="black"
)

# label values
for i, v in enumerate(coef_df["Coefficient"]):
    offset = 0.03 if v >= 0 else -0.03
    ax.text(v + offset, i, f"{v:.3f}", fontsize=11, va="center")

plt.title("Linear Regression Feature Influence (Scaled)", fontsize=20, weight="bold", pad=20)
plt.xlabel("Coefficient Value")
plt.ylabel("Feature")
ax.xaxis.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.show()


# Actual vs Predicted
plt.figure(figsize=(8, 8))
sns.scatterplot(
    x=y_test[:10000],
    y=y_pred[:10000],
    alpha=0.3
)

# perfect fit line
low = min(y_test[:10000].min(), y_pred[:10000].min())
high = max(y_test[:10000].max(), y_pred[:10000].max())

plt.plot([low, high], [low, high], "r--", label="Perfect Fit Line")

plt.xlabel("Actual Global Active Power")
plt.ylabel("Predicted Global Active Power")
plt.title("Linear Regression: Actual vs Predicted (Time-Aware Split)")
plt.grid(True, linestyle="--", alpha=0.4)
plt.legend()
plt.tight_layout()
plt.show()


# Residuals vs Predictions
residuals = y_test.values - y_pred

plt.figure(figsize=(10, 5))
sns.scatterplot(
    x=y_pred[:20000],
    y=residuals[:20000],
    alpha=0.3
)

plt.axhline(0, color="red", linestyle="--")
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.title("Linear Regression Residuals vs Predictions (Time-Aware Split)")
plt.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.show()


# Residual Distribution
plt.figure(figsize=(10, 5))
sns.histplot(residuals, bins=50, kde=True, color="steelblue")

plt.title("Linear Regression Residual Distribution")
plt.xlabel("Residual")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

