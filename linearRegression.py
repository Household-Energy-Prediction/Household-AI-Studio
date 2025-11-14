import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# load final dataset
df = pd.read_csv("final_data.csv", index_col = "datetime", parse_dates = True)
df = df.dropna()

# features and target
X = df[["Voltage",
        "Global_reactive_power",
        "Sub_metering_1",
        "Sub_metering_2",
        "Sub_metering_3"]]

y = df["Global_active_power"]

# split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size = 0.2, random_state = 42
)

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)

# scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# train LR
lin_reg = LinearRegression()
lin_reg.fit(X_train_scaled, y_train)

# predict
y_pred = lin_reg.predict(X_test_scaled)

# evaluate
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nLinear Regression Results (Scaled)")
print(f"MAE:  {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R²:   {r2:.4f}")

# coefficients
coef_df = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": lin_reg.coef_
}).sort_values(by = "Coefficient", ascending = False)

plt.figure(figsize = (6,4))
plt.barh(coef_df["Feature"], coef_df["Coefficient"])
plt.xlabel("Coefficient Value")
plt.title("Feature Influence (Linear Regression)")
plt.tight_layout()
plt.show()

