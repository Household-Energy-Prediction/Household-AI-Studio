import pandas as pd
import numpy as np

def resample(df):
    resampled_df = df.copy()

    # Tesample hourly
    hourly = resampled_df.resample('H').mean()
    # Resample daily
    daily = resampled_df.resample('D').mean()
    # Resample weekly
    weekly = resampled_df.resample('W').mean()
    # Resample monthly 
    monthly = resampled_df.resample('M').mean()
    # Resample  quarterly
    quarterly = resampled_df.resample('Q').mean()

    print(f"Original data: {resampled_df.shape}")
    print(f"Hourly data: {hourly.shape}")
    print(f"Daily data: {daily.shape}")
    print(f"Weekly data: {weekly.shape}")
    print(f"Monthly data: {monthly.shape}")
    print(f"Quarterly data: {quarterly.shape}")

    print(resampled_df.head())
    print(hourly.head())
    print(daily.head())
    print(weekly.head())
    print(monthly.head())
    print(quarterly.head())

    hourly.to_csv('hourly.csv')
    daily.to_csv('daily.csv')
    weekly.to_csv('weekly.csv')
    monthly.to_csv('monthly.csv')
    quarterly.to_csv('quarterly.csv')

    return (df, hourly, daily, weekly, monthly, quarterly)