import pandas as pd

def add_durations_set_negatives_to_zero():

    dataFrame = pd.read_csv("../../data/charging_sessions_cleaned.csv", parse_dates=["connectionTime", "disconnectTime", "doneChargingTime"])

    #set doneChargingTime = connectionTime when doneChargingTime < connectionTime (result -> durationUntilFullCharge = 0 when initially negative)
    dataFrame.loc[dataFrame["doneChargingTime"] < dataFrame["connectionTime"], "doneChargingTime"] = dataFrame["connectionTime"]

    #set disconnectTime = connectionTime when disconnectTime < connectionTime (result -> duration = 0 when initially negative)
    dataFrame.loc[dataFrame["disconnectTime"] < dataFrame["connectionTime"], "disconnectTime"] = dataFrame["connectionTime"]

    #add "duation" column: whole time span the EV is CONNECTED
    dataFrame["duration"] = (dataFrame["disconnectTime"] - dataFrame["connectionTime"])

    #add "doneChargingDuration" column: whole time span the EV is CHARGING
    dataFrame["durationUntilFullCharge"] = (dataFrame["doneChargingTime"] - dataFrame["connectionTime"])

    #save to csv
    dataFrame.to_csv("../../data/charging_sessions_cleaned.csv", index=False)





add_durations_set_negatives_to_zero()
