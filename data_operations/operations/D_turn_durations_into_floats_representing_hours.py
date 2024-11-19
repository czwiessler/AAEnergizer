import pandas as pd

def turn_durations_into_floats_representing_hours():

    dataFrame = pd.read_csv("data/charging_sessions_cleaned.csv", parse_dates=["connectionTime", "disconnectTime", "doneChargingTime"])

    dataFrame["duration"] = pd.to_timedelta(dataFrame["duration"])

    dataFrame["durationUntilFullCharge"] = pd.to_timedelta(dataFrame["durationUntilFullCharge"])

    #turn nano second data type into float representing hours
    dataFrame["durationUntilFullCharge"] = dataFrame["durationUntilFullCharge"].dt.total_seconds() / 3600
    dataFrame["duration"] = dataFrame["duration"].dt.total_seconds() / 3600

    #save to csv
    dataFrame.to_csv("data/charging_sessions_cleaned.csv", index=False)




turn_durations_into_floats_representing_hours()
