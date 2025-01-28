import pandas as pd

def turn_durations_into_floats_representing_hours(dataset_path):

    dataFrame = pd.read_csv(dataset_path, parse_dates=["connectionTime", "disconnectTime", "doneChargingTime"])

    dataFrame["duration"] = pd.to_timedelta(dataFrame["duration"])

    dataFrame["durationUntilFullCharge"] = pd.to_timedelta(dataFrame["durationUntilFullCharge"])

    #turn nano second data type into float representing hours
    dataFrame["durationUntilFullCharge"] = dataFrame["durationUntilFullCharge"].dt.total_seconds() / 3600
    dataFrame["duration"] = dataFrame["duration"].dt.total_seconds() / 3600

    #save to csv
    dataFrame.to_csv(dataset_path, index=False)

    print("D_: durations turned into floats representing hours")


if __name__ == "__main__":
    turn_durations_into_floats_representing_hours(dataset_path="data/processed/charging_sessions_cleaned.csv")
