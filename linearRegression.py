import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split

# load preprocessed dataset 
df = pd.read_csv("data_with_datetime.csv", index_col="datetime", parse_dates=True)

# Drop missing values (same across both models)
df = df.dropna()

# define fetures and target
X = df[["Voltage",
        "Global_reactive_power",
        "Sub_metering_1",
        "Sub_metering_2",
        "Sub_metering_3"]]

y = df["Global_active_power"]

# split data 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)

# train Linear Regression model
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

# Predict
y_pred = lin_reg.predict(X_test)

# 5. Model evaluation
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nLinear Regression Results")
print(f"MAE:  {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R²:   {r2:.4f}")

# 6. Plot actual vs predicted
plt.figure(figsize=(8,8))
plt.scatter(y_test[:5000], y_pred[:5000], alpha=0.3)
min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--')
plt.xlabel("Actual Global Active Power")
plt.ylabel("Predicted Global Active Power")
plt.title("Actual vs Predicted (Linear Regression)")
plt.grid(True)
plt.show()

# 7. Plot feature coefficients
coef_df = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": lin_reg.coef_
}).sort_values(by="Coefficient", ascending=False)

plt.figure(figsize=(6,4))
plt.barh(coef_df["Feature"], coef_df["Coefficient"])
plt.xlabel("Coefficient Value")
plt.title("Feature Influence (Linear Regression)")
plt.tight_layout()
plt.show()
