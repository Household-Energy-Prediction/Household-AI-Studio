import pandas as pd
import zipfile
import os

print("Convert to datetime")

zip_path = "household_power_consumption.csv.zip"
csv_name = "household_power_consumption.csv"

# Ensure CSV is extracted
if not os.path.exists(csv_name):
    if os.path.exists(zip_path):
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(".")
        print(f"Extracted {csv_name}")
    else:
        raise FileNotFoundError(
            f"Could not find {csv_name} or {zip_path} in the current folder."
        )

print("Loading CSV...")
df = pd.read_csv(csv_name, sep = ",", low_memory = False)

# Combine Date and Time into datetime
print("Converting Date + Time to datetime...")
df["datetime"] = pd.to_datetime(
    df["Date"] + " " + df["Time"],
    format = "%d/%m/%Y %H:%M:%S",
    errors = "coerce"
)

# Drop old columns
df.drop(columns = ["Date", "Time"], inplace = True)

# Convert measurement columns to numeric
numeric_cols = [
    "Global_active_power",
    "Global_reactive_power",
    "Voltage",
    "Global_intensity",
    "Sub_metering_1",
    "Sub_metering_2",
    "Sub_metering_3",
]
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors = "coerce")

# Set datetime as index
df.set_index("datetime", inplace = True)

# Save the cleaned dataset
df.to_csv("data_with_datetime.csv", index = True)

print("data_with_datetime.csv created")
print(df.info())
