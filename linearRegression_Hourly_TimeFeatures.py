import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler

# LOAD DATA (Hourly + Time Features)
df = pd.read_csv("final_data_hourly_timeFeatures.csv", index_col="datetime", parse_dates=True)

print("Loaded dataset shape:", df.shape)
print("Date range:", df.index.min(), "→", df.index.max())

# TRAIN/TEST SPLIT (TIME-AWARE)
train_df = df.iloc[:int(len(df)*0.8)]
test_df = df.iloc[int(len(df)*0.8):]

print("Train size:", len(train_df), " Test size:", len(test_df))
print("Train range:", train_df.index.min(), "→", train_df.index.max())
print("Test range: ", test_df.index.min(), "→", test_df.index.max())

# DEFINE FEATURES & TARGET
target = "Global_active_power"

features = [
    "Voltage", "Global_reactive_power",
    "Sub_metering_1", "Sub_metering_2", "Sub_metering_3",
    "hour", "dayofweek", "month", "year",
    "is_weekend",
    "hour_sin", "hour_cos",
    "dow_sin", "dow_cos",
    "month_sin", "month_cos"
]

X_train = train_df[features]
X_test = test_df[features]
y_train = train_df[target]
y_test = test_df[target]

# SCALE FEATURES
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# TRAIN MODEL
lin_reg = LinearRegression()
lin_reg.fit(X_train_scaled, y_train)

# PREDICT + EVALUATE
y_pred = lin_reg.predict(X_test_scaled)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\nLinear Regression (Hourly + Time Features)")
print(f"MAE:  {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R²:   {r2:.4f}")

# FEATURE COEFFICIENTS
coef_df = pd.DataFrame({
    "Feature": features,
    "Coefficient": lin_reg.coef_
})

coef_df["AbsCoef"] = coef_df["Coefficient"].abs()
coef_df = coef_df.sort_values("AbsCoef", ascending=True)

plt.figure(figsize=(12, 7))
sns.barplot(
    x="Coefficient",
    y="Feature",
    data=coef_df,
    palette="Blues",
    edgecolor="black"
)
plt.title("Linear Regression Feature Influence (Hourly + Time Features)")
plt.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()
plt.show()

# ACTUAL VS PREDICTED
plt.figure(figsize=(8, 8))
sns.scatterplot(x=y_test[:4000], y=y_pred[:4000], alpha=0.3)
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         'r--', label="Perfect Fit")
plt.title("Linear Regression (Hourly + Time Features): Actual vs Predicted")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.grid(True, linestyle='--', alpha=0.4)
plt.legend()
plt.tight_layout()
plt.show()

# RESIDUAL PLOTS
residuals = y_test - y_pred

plt.figure(figsize=(10, 5))
sns.scatterplot(x=y_pred[:4000], y=residuals[:4000], alpha=0.3)
plt.axhline(0, color='red', linestyle='--')
plt.title("Residuals vs Predictions (LR Hourly + Time Features)")
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 5))
sns.histplot(residuals, bins=40, kde=True, color="steelblue")
plt.title("Residual Distribution (LR Hourly + Time Features)")
plt.xlabel("Residual")
plt.tight_layout()
plt.show()
