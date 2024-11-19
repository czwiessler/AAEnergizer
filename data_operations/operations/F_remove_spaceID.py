import pandas as pd

def remove_spaceID():
    dataFrame = pd.read_csv("data/charging_sessions_cleaned.csv", parse_dates=["connectionTime", "disconnectTime", "doneChargingTime"])

    # Remove column stationID
    dataFrame.drop(columns=["spaceID"], inplace=True)

    # Save to csv
    dataFrame.to_csv("data/charging_sessions_cleaned.csv", index=False)



remove_spaceID()