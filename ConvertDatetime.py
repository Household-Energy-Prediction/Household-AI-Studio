import pandas as pd

def convertDatetime(df):
    converted_df = df.copy()
    converted_df["datetime"] = pd.to_datetime(
        converted_df["Date"] + " " + converted_df["Time"],
        format = "%d/%m/%Y %H:%M:%S",
        errors = "coerce"
    )
    # Drop old columns
    converted_df.drop(columns = ["Date", "Time"], inplace = True)

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
        converted_df[col] = pd.to_numeric(converted_df[col], errors = "coerce")

    # Set datetime as index
    converted_df.set_index("datetime", inplace = True)

    # Save the cleaned dataset
    converted_df.to_csv("data_with_datetime.csv", index = True)

    return converted_df;

if __name__ == "__main__":
    import pandas as pd
    df = pd.read_csv("household_power_consumption.csv/household_power_consumption.csv", sep=',')
    cleaned_df = convertDatetime(df)
    print("data_with_datetime.csv created successfully!")
    print(cleaned_df.head())
