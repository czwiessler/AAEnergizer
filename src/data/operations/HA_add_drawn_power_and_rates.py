import pandas as pd


def add_drawn_power_and_rates(nn_dataset_path, dataset_path):

    #session data set for retrieval of unique station ID counts
    session_data = pd.read_csv(dataset_path)

    #get the number of unique stationIDs for each siteID within session data
    site_1_total_stations = session_data[session_data['siteID'] == 1]['stationID'].nunique()
    site_2_total_stations = session_data[session_data['siteID'] == 2]['stationID'].nunique()

    df = pd.read_csv(nn_dataset_path)

    # Add 'total_drawn_power' columns
    df['total_drawn_power_1'] = df['chargingSessions_site_1'] * df['avgChargingPower_site_1']
    df['total_drawn_power_2'] = df['chargingSessions_site_2'] * df['avgChargingPower_site_2']

    # Add 'utilization_rate' columns
    df['utilization_rate_1'] = df['activeSessions_site_1'] / site_1_total_stations
    df['utilization_rate_2'] = df['activeSessions_site_2'] / site_2_total_stations

    # Add 'idle_rate' columns
    df['idle_rate_1'] = (df['activeSessions_site_1'] - df['chargingSessions_site_1']) / df['activeSessions_site_1']
    df['idle_rate_2'] = (df['activeSessions_site_2'] - df['chargingSessions_site_2']) / df['activeSessions_site_2']

    # Handle potential division by zero for idle_rate
    df['idle_rate_1'] = df['idle_rate_1'].fillna(0)  # If activeSessions_site_1 is 0
    df['idle_rate_2'] = df['idle_rate_2'].fillna(0)  # If activeSessions_site_2 is 0

    df.to_csv(nn_dataset_path, index=False)
    print("KPI columns added to the hourly data set and saved to ", nn_dataset_path)

    return df


if __name__ == "__main__":
    add_drawn_power_and_rates(nn_dataset_path="data/processed/hourly_avg_power.csv", dataset_path="data/processed/charging_sessions_cleaned.csv")