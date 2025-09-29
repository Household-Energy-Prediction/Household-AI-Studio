import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

def normalize(df):
    normalized_df = df.copy()
    numeric_cols = [
        "Global_active_power",
        "Global_reactive_power",
        "Voltage",
        "Global_intensity",
        "Sub_metering_1",
        "Sub_metering_2",
        "Sub_metering_3",
    ]
    normalized_df[numeric_cols] = scaler.fit_transform(normalized_df[numeric_cols])

    return normalized_df;