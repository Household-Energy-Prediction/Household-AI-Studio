import pandas as pd
import numpy as np
import seaborn as sns
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
})

# Sort by absolute value (importance)
coef_df["AbsCoef"] = coef_df["Coefficient"].abs()
coef_df = coef_df.sort_values(by="AbsCoef", ascending=True)

# Plot 
plt.figure(figsize=(12, 7))
sns.set(style="white")

ax = sns.barplot(
    x="Coefficient",
    y="Feature",
    data=coef_df,
    hue="Feature",
    palette="Blues",
    legend=False,     # Prevents extra legend box
    linewidth=1,
    edgecolor="black"
)


# Add value labels
for i, v in enumerate(coef_df["Coefficient"]):
    ax.text(
        v + (0.02 if v >= 0 else -0.02),   # offset for direction
        i,
        f"{v:.3f}",
        color="black",
        va="center",
        fontsize=11,
        fontweight="bold"
    )

# Title & labels
plt.title("Linear Regression Feature Influence (Scaled)", fontsize=20, weight="bold", pad=20)
plt.xlabel("Coefficient Value", fontsize=14)
plt.ylabel("Feature", fontsize=14)

# Grid lines
ax.xaxis.grid(True, linestyle='--', alpha=0.4)
ax.yaxis.grid(False)

plt.tight_layout()
plt.show()

