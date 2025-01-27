import pandas as pd

def add_durations_set_negatives_to_zero(dataset_path):

    dataFrame = pd.read_csv(dataset_path, parse_dates=["connectionTime", "disconnectTime", "doneChargingTime"])

    #set doneChargingTime = connectionTime when doneChargingTime < connectionTime (result -> durationUntilFullCharge = 0 when initially negative)
    dataFrame.loc[dataFrame["doneChargingTime"] < dataFrame["connectionTime"], "doneChargingTime"] = dataFrame["connectionTime"]

    # gibts glaub ich garnet...
    #set disconnectTime = connectionTime when disconnectTime < connectionTime (result -> duration = 0 when initially negative)
    dataFrame.loc[dataFrame["disconnectTime"] < dataFrame["connectionTime"], "disconnectTime"] = dataFrame["connectionTime"]

    #...dafür gibts aber disconnectTime < doneChargingTime
    # set disconnectTime = doneChargingTime when disconnectTime < doneChargingTime (result -> duration = 0 when initially negative)
    dataFrame.loc[dataFrame["disconnectTime"] < dataFrame["doneChargingTime"], "doneChargingTime"] = dataFrame["disconnectTime"]

    #add "duration" column: whole time span the EV is CONNECTED
    dataFrame["duration"] = (dataFrame["disconnectTime"] - dataFrame["connectionTime"])

    #add "doneChargingDuration" column: whole time span the EV is CHARGING
    dataFrame["durationUntilFullCharge"] = (dataFrame["doneChargingTime"] - dataFrame["connectionTime"])

    #save to csv
    dataFrame.to_csv(dataset_path, index=False)



if __name__ == "__main__":
    add_durations_set_negatives_to_zero(dataset_path="data/processed/charging_sessions_cleaned.csv")
