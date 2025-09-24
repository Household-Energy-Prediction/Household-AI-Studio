import pandas as pd
import numpy as np

df = pd.read_csv("household_power_consumption.csv")

# Check columns to see which variables can be added
print("Dataframe columns:", df.columns)
print("First 5 rows of the dataframe: ")
print(df.head(5))

# Create a new column named isWeekend that has 1 if the date is a weekend, 0 if not
df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
df["is_weekend"] = df["Date"].dt.dayofweek >= 5

# Check to see if it has been made
print("New columns: ", df.columns)
print("First 5 rows of the dataframe with new variable: ")
print(df.head(5))

# Create a new column named Season that has winter, spring, summer, or fall depending on the season
# Create a season getter from the month
def seasons(date):
    month = date.month
    if (month == 12 or month == 1 or month == 2):
        return "Winter"
    elif (month == 3 or month == 4 or month == 5):
        return "Spring"
    elif (month == 6 or month == 7 or month == 8):
        return "Summer"
    else:
        return "Fall"
    
# Apply the function to the Date column and put the results in the Season column
df["Season"] = df["Date"].apply(seasons)

print("New columns: ", df.columns)
print("First 5 columns of the dataframe with new variable: ")
print(df.head(5))
