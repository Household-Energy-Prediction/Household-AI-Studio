# %%
import pandas as pd 
import numpy as np
# %% 
df = pd.read_csv("/Users/kimayajadhav/Desktop/BTTAI/household_power_consumption.txt", sep = ";", na_values = "?")
print("Dataframe shape:", df.shape)
df.head()
# %%
# Looking for missing values
nan_count = np.sum(df.isnull(), axis = 0)
nan_count
# %% 

# Dropping rows with missing values 
df = df.dropna()
print("New shape of dataframe after dropping missing values:", df.shape)
# %%
updated_nan_count = np.sum(df.isnull(), axis = 0)
updated_nan_count
# %%

# Searching for outliers 

features = ["Global_active_power", "Global_reactive_power", "Voltage", 
            "Global_intensity", "Sub_metering_1", "Sub_metering_2", "Sub_metering_3"]

for col in features:
    z_scores = (df[col] - df[col].mean()) / df[col].std()
    outliers_flag = np.abs(z_scores) > 3   # boolean mask
    outliers_df = df[outliers_flag]        # filter directly
    print(f"{col} → Total outlier rows: {len(outliers_df)}")

    


# %%
