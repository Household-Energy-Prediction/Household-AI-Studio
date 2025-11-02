from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


X = df[["Voltage", "Global_reactive_power", "Sub_metering_1", "Sub_metering_2", "Sub_metering_3"]]
y = df["Global_active_power"]

#test 20 --80 train
X_train, X_test, y_train, y_test = train_test_split(  X, y, test_size=0.2, random_state=42)

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)

# data train and predict
LinearR = LinearRegression()
LinearR.fit(X_train, y_train)
y_pred = LinearR.predict(X_test)

# model evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)

print(f"Mean Absolute Error(MAE): {mae:.3f}")
print(f"Root Mean Square Error(RMSE): {rmse:.3f}")
print(f"R2: {r2:.3f}")
#plot
plt.figure(figsize=(8,8))
plt.scatter(y_test[:10000], y_pred[:10000], alpha=0.5) #only the first 10000 for each as with all of the points it looks like there is a point everywhere.
min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', label="Fit line")
plt.xlabel("Actual Global Active Power")
plt.ylabel("Predicted Global Active Power")
plt.title("Actual vs Predicted Linear Regression")
plt.grid(True)
plt.show()

# coefficient plot for feature influence
coef_df = pd.DataFrame({"Feature": X.columns, "coef": LinearR.coef_}).sort_values(by="coef", ascending=False)

plt.figure(figsize=(4,6))
plt.barh(coef_df["Feature"], coef_df["coef"])
plt.title("Feature Influence")
plt.xlabel("Coefficient Value")
plt.tight_layout()
plt.show()
