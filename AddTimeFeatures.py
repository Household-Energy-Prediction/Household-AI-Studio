import pandas as pd
import numpy as np

# Load hourly dataset
df = pd.read_csv("final_data_hourly.csv", index_col="datetime", parse_dates=True)

print("Loaded hourly data:", df.shape)

# Extract Raw Time Components
df["hour"] = df.index.hour
df["dayofweek"] = df.index.dayofweek   # Monday=0 ... Sunday=6
df["month"] = df.index.month
df["year"] = df.index.year
df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)

# Add Cyclical Time Encodings
# Daily cycle (24 hours)
df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

# Weekly cycle (7 days)
df["dow_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
df["dow_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)

# Yearly cycle (12 months)
df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

# Save the enhanced dataset
output_file = "final_data_hourly_timeFeatures.csv"
df.to_csv(output_file)

print("\n✔ Saved:", output_file)
print("New dataset shape:", df.shape)
print("\nPreview:")
print(df.head())
