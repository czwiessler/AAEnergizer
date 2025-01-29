import pandas as pd


def remove_duration_outliers(dataset_path):
    dataFrame = pd.read_csv(dataset_path, parse_dates=["connectionTime", "disconnectTime", "doneChargingTime"])

    # Ensure 'duration' column is numeric
    dataFrame['duration'] = pd.to_numeric(dataFrame['duration'], errors='coerce')

    # .997th quantile is around 30h
    duration_threshhold = dataFrame['duration'].quantile(0.997)

    # Filter out rows where 'duration' exceeds the 0.997 quantile
    dataFrame = dataFrame[dataFrame['duration'] <= duration_threshhold]

    # save to csv
    dataFrame.to_csv(dataset_path, index=False)

    print("DA_: duration outliers removed")


if __name__ == "__main__":
    remove_duration_outliers(dataset_path="data/processed/charging_sessions_cleaned.csv")