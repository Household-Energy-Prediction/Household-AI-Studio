import pandas as pd 
import numpy as np

def removeOutliers(df):
    removed_df = df.copy()

    features = ["Global_active_power", "Global_reactive_power", "Voltage", 
            "Global_intensity", "Sub_metering_1", "Sub_metering_2", "Sub_metering_3"]

    for col in features:
        z_scores = (removed_df[col] - removed_df[col].mean()) / removed_df[col].std()
        outliers_flag = np.abs(z_scores) > 3   # boolean mask
        outliers_df = removed_df[outliers_flag]        # filter directly
        print(f"{col} → Total outlier rows: {len(outliers_df)}")
        removed_df = removed_df.drop(outliers_df.index)
    
    return removed_df;
    
