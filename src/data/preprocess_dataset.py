from winreg import HKEY_CLASSES_ROOT

import src.data.operations.A_copy_csv as A_copy_csv
import src.data.operations.B_remove_single_rows_with_implausible_timestamps as B_remove_single_rows_with_implausible_timestamps
import src.data.operations.C_add_durations_set_negatives_to_zero as C_add_durations_set_negatives_to_zero
import src.data.operations.D_turn_durations_into_floats_representing_hours as D_turn_durations_into_floats_representing_hours
import src.data.operations.E_add_charging_power_remove_outliers as E_add_charging_power_remove_outliers
import src.data.operations.F_remove_spaceID as F_remove_spaceID
import src.data.operations.G_add_utilization as G_add_utilization
import src.data.operations.H_create_nn_ds as H_create_nn_ds
import src.data.operations.I_prepare_nn_ds as I_prepare_nn_ds
import src.data.operations.Z_cut_ds as Z_cut_ds
import pandas as pd

def preprocess_dataset():
    raw_dataset_path = "data/raw/charging_sessions.csv"
    raw_weather_dataset_path = "data/raw/weather_burbank_airport.csv"
    processed_dataset_path = "data/processed/charging_sessions_cleaned.csv"
    nn_dataset_path = "data/processed/hourly_avg_power.csv"

    A_copy_csv.copy_csv(from_path=raw_dataset_path, to_path=processed_dataset_path)
    B_remove_single_rows_with_implausible_timestamps.remove_single_rows_with_implausible_timestamps(dataset_path=processed_dataset_path)
    C_add_durations_set_negatives_to_zero.add_durations_set_negatives_to_zero(dataset_path=processed_dataset_path)
    D_turn_durations_into_floats_representing_hours.turn_durations_into_floats_representing_hours(dataset_path=processed_dataset_path)
    E_add_charging_power_remove_outliers.add_charging_power_remove_outliers(dataset_path=processed_dataset_path)
    F_remove_spaceID.remove_spaceID(dataset_path=processed_dataset_path)

    #G_add_utilization.add_utilization(dataset_path=processed_dataset_path)

    H_create_nn_ds.create_nn_ds(dataset_path=processed_dataset_path,
                                weather_dataset_path=raw_weather_dataset_path,
                                nn_dataset_path=nn_dataset_path)

    I_prepare_nn_ds.prepare_nn_ds(dataset_path=nn_dataset_path)

    Z_cut_ds.cut_ds(dataset_path=nn_dataset_path, start_date="2018-09-05", end_date="2020-03-01")

    print("Preprocessing done.")

    # load the data set
    df = pd.read_csv(processed_dataset_path)
    return df



if __name__ == "__main__":
    preprocess_dataset()

