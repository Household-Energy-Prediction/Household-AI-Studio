import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
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

# feature importance plot
plt.figure(figsize = (8, 4))
plt.barh(X.columns, rf.feature_importances_)
plt.xlabel("Feature Importance")
plt.title("Random Forest Feature Importance (Tuned)")
plt.tight_layout()
plt.show()

