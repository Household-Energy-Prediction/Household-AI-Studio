import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

# Load hourly data
df = pd.read_csv("final_data_hourly.csv", index_col="datetime", parse_dates=True)
df = df.dropna()

# Features
X = df[["Voltage", "Global_reactive_power",
        "Sub_metering_1", "Sub_metering_2", "Sub_metering_3"]]

y = df["Global_active_power"]

# Time-aware split: 80% train, 20% test
split_idx = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

print("Train size:", len(X_train), "Test size:", len(X_test))

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Linear Regression
lin_reg = LinearRegression()
lin_reg.fit(X_train_scaled, y_train)

# Predictions
y_pred = lin_reg.predict(X_test_scaled)

# Evaluation
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\nLinear Regression (Hourly, Time-Aware Split) Results")
print(f"MAE:  {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R²:   {r2:.4f}")

# Coefficients
coef_df = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": lin_reg.coef_
}).sort_values(by="Coefficient")

# Plot Coefficients
plt.figure(figsize=(12, 7))
sns.barplot(x="Coefficient", y="Feature", data=coef_df,
            palette="Blues", edgecolor="black")
plt.title("Linear Regression Feature Influence (Hourly)", fontsize=18)
plt.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()
plt.show()

# Plot Actual vs Predicted
plt.figure(figsize=(8, 8))
sns.scatterplot(x=y_test[:5000], y=y_pred[:5000], alpha=0.3)
min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', label="Perfect Fit")
plt.title("Linear Regression (Hourly): Actual vs Predicted")
plt.xlabel("Actual Global Active Power")
plt.ylabel("Predicted Global Active Power")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()
plt.show()

# Residuals
residuals = y_test - y_pred

plt.figure(figsize=(10, 5))
sns.scatterplot(x=y_pred, y=residuals, alpha=0.3)
plt.axhline(0, color="red", linestyle="--")
plt.title("Linear Regression (Hourly): Residuals vs Predictions")
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()
plt.show()

# Residual Distribution
plt.figure(figsize=(10, 5))
sns.histplot(residuals, bins=50, kde=True)
plt.title("Linear Regression (Hourly): Residual Distribution")
plt.xlabel("Residual")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()
