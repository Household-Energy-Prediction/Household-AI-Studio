import pandas as pd
import numpy as np

df = pd.read_csv("households.txt", sep=";")

# Convert to datetime 
df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%d/%m/%Y %H:%M:%S')
df = df.drop(['Date', 'Time'], axis=1)
df = df.set_index('DateTime')


# Converting numeric column to float
numeric_cols = ['Global_active_power', 'Global_reactive_power', 'Voltage', 
                'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']
df[numeric_cols] = df[numeric_cols].astype(float)

# Tesample hourly
hourly = df.resample('H').mean()
# Resample daily
daily = df.resample('D').mean()
# Resample weekly
weekly = df.resample('W').mean()
# Resample monthly 
monthly = df.resample('M').mean()
# Resample  quarterly
quarterly = df.resample('Q').mean()

print(f"Original data: {df.shape}")
print(f"Hourly data: {hourly.shape}")
print(f"Daily data: {daily.shape}")
print(f"Weekly data: {weekly.shape}")
print(f"Monthly data: {monthly.shape}")
print(f"Quarterly data: {quarterly.shape}")

print(df.head())
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
