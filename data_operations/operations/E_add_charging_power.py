import pandas as pd

def add_charging_power():
    dataFrame = pd.read_csv("data/charging_sessions_cleaned.csv", parse_dates=["connectionTime", "disconnectTime", "doneChargingTime"])

    # Add a new column chargingPower (Ladeleistung in kW)
    dataFrame["chargingPower"] = .0

    # Case 1: Calculate chargingPower where durationUntilFullCharge > 0
    mask_full_charge = dataFrame["durationUntilFullCharge"] > 0
    dataFrame.loc[mask_full_charge, "chargingPower"] = (
        dataFrame["kWhDelivered"] / dataFrame.loc[mask_full_charge, "durationUntilFullCharge"]
    )

    # Case 2: Calculate chargingPower using duration if doneChargingTime is null
    mask_null_done_charging = dataFrame["doneChargingTime"].isna()
    dataFrame.loc[mask_null_done_charging, "chargingPower"] = (
        dataFrame["kWhDelivered"] / dataFrame.loc[mask_null_done_charging, "duration"]
    )


    #handling, damit die Zeit in der nicht geladen wird nicht die Berechnung der Ladeleistung verfälscht


    #save to csv
    dataFrame.to_csv("data/charging_sessions_cleaned.csv", index=False)



add_charging_power()