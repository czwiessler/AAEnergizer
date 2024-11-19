import data_operations.operations.A_copy_csv as A_copy_csv
import data_operations.operations.B_remove_two_big_time_difference_rows as B_remove_two_big_time_difference_rows
import data_operations.operations.C_add_durations_set_negatives_to_zero as C_add_durations_set_negatives_to_zero
import data_operations.operations.D_turn_durations_into_floats_representing_hours as D_turn_durations_into_floats_representing_hours
import data_operations.operations.E_add_charging_power as E_add_charging_power
import data_operations.operations.F_remove_spaceID as F_remove_spaceID
import pandas as pd

def preprocess_dataset():
    A_copy_csv.copy_csv()
    B_remove_two_big_time_difference_rows.remove_two_big_time_difference_rows()
    C_add_durations_set_negatives_to_zero.add_durations_set_negatives_to_zero()
    D_turn_durations_into_floats_representing_hours.turn_durations_into_floats_representing_hours()
    E_add_charging_power.add_charging_power()
    F_remove_spaceID.remove_spaceID()

    print("Preprocessing done.")

    # load the data set
    df = pd.read_csv("data/charging_sessions_cleaned.csv")
    return df

preprocess_dataset()

