import pandas as pd

def add_charging_power_remove_outliers(dataset_path):
    dataFrame = pd.read_csv(dataset_path, parse_dates=["connectionTime", "disconnectTime", "doneChargingTime"])

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

    ##############################################################################################################
    #remove outliers with chargingpower > 50kW
    ids_to_be_removed = [
        '5bc9297df9af8b0dc677c7cf',
        '61341051f9af8b434b144cd3',
        '60cd3a8ff9af8b228751eabf',
        '610c829df9af8b0580e9872c',
        '5dccae31f9af8b1ddbaaddf2',
        '610c829df9af8b0580e98731',
        '5be2fd24f9af8b2b0edfa126',
        '5bc918bff9af8b0dc677b99d',
        '5bc917d0f9af8b0dc677b8cf',
        '610c829df9af8b0580e9873f',
        '610c829df9af8b0580e9873e',
        '5df82ffbf9af8b399f100ec4',
        '5df82ffbf9af8b399f100ec5',
        '60cd3a8ff9af8b228751eac1',
        '61171011f9af8b160784513b',
        '5d35038df9af8b5eb4734d24',
        '610c829df9af8b0580e98737'
    ]

    # remove all rows from ids_to_be_removed from df
    dataFrame = dataFrame[~dataFrame['id'].isin(ids_to_be_removed)]

    ##############################################################################################################

    #save to csv
    dataFrame.to_csv(dataset_path, index=False)



if __name__ == "__main__":
    add_charging_power_remove_outliers(dataset_path="data/processed/charging_sessions_cleaned.csv")