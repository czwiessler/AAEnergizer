import pandas as pd

def remove_spaceID(dataset_path):
    dataFrame = pd.read_csv(dataset_path, parse_dates=["connectionTime", "disconnectTime", "doneChargingTime"])

    # Remove column stationID
    dataFrame.drop(columns=["spaceID"], inplace=True)

    # Save to csv
    dataFrame.to_csv(dataset_path, index=False)

    print("F_: Removed spaceID column")


if __name__ == "__main__":
    remove_spaceID(dataset_path="data/processed/charging_sessions_cleaned.csv")