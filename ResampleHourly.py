import pandas as pd

# Load the final cleaned minute-level dataset
df = pd.read_csv("final_data.csv", index_col="datetime", parse_dates=True)

print("Original data shape:", df.shape)
print("Original date range:", df.index.min(), "→", df.index.max())

# Resample to HOURLY AVERAGES
df_hourly = df.resample("1H").mean()

print("\nHourly data shape:", df_hourly.shape)
print("Hourly date range:", df_hourly.index.min(), "→", df_hourly.index.max())

# Drop rows with missing values after resampling
df_hourly = df_hourly.dropna()

print("\nAfter dropna:", df_hourly.shape)

# Save the hourly dataset
df_hourly.to_csv("final_data_hourly.csv")

print("\n✔ Saved: final_data_hourly.csv")
