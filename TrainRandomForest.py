import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt

# load preprocessed dataset 
df = pd.read_csv("data_with_datetime.csv", index_col="datetime", parse_dates=True)
df = df.sample(frac=0.05, random_state=42)  # use 5% of the data

# drop missing values 
df = df.dropna()

# features and target defined
X = df[["Global_reactive_power", "Voltage", "Global_intensity",
        "Sub_metering_1", "Sub_metering_2", "Sub_metering_3"]]
y = df["Global_active_power"]

# data split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# create + train Random Forest
rf = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
rf.fit(X_train, y_train)

# model pediction
y_pred = rf.predict(X_test)

# performance evaluation
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n Random Forest Model Result")
print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R²: {r2:.4f}")

# plot feature importance
plt.figure(figsize=(8, 4))
plt.barh(X.columns, rf.feature_importances_)
plt.xlabel("Feature Importance")
plt.title("Random Forest Feature Importance")
plt.tight_layout()
plt.show()
