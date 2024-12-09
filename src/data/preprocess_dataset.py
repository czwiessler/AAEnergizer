import src.data.operations.A_copy_csv as A_copy_csv
import src.data.operations.B_remove_two_big_time_difference_rows as B_remove_two_big_time_difference_rows
import src.data.operations.C_add_durations_set_negatives_to_zero as C_add_durations_set_negatives_to_zero
import src.data.operations.D_turn_durations_into_floats_representing_hours as D_turn_durations_into_floats_representing_hours
import src.data.operations.E_add_charging_power as E_add_charging_power
import src.data.operations.F_remove_spaceID as F_remove_spaceID
import pandas as pd

def preprocess_dataset():
    raw_dataset_path = "data/raw/charging_sessions.csv"
    processed_dataset_path = "data/processed/charging_sessions_cleaned.csv"

    A_copy_csv.copy_csv(from_path=raw_dataset_path, to_path=processed_dataset_path)
    B_remove_two_big_time_difference_rows.remove_two_big_time_difference_rows(dataset_path=processed_dataset_path)
    C_add_durations_set_negatives_to_zero.add_durations_set_negatives_to_zero(dataset_path=processed_dataset_path)
    D_turn_durations_into_floats_representing_hours.turn_durations_into_floats_representing_hours(dataset_path=processed_dataset_path)
    E_add_charging_power.add_charging_power(dataset_path=processed_dataset_path)
    F_remove_spaceID.remove_spaceID(dataset_path=processed_dataset_path)

    print("Preprocessing done.")

    # load the data set
    df = pd.read_csv(processed_dataset_path)
    return df



if __name__ == "__main__":
    preprocess_dataset()

