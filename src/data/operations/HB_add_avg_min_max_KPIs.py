import pandas as pd

def add_avg_min_max_KPIs(nn_dataset_path, KPI_dataset_path):


    hourly_data = pd.read_csv(nn_dataset_path)

    #parse
    hourly_data['hour'] = pd.to_datetime(hourly_data['hour'])

    #to get full days going from 00:00:00 to 23:00:00
    hourly_data = hourly_data.iloc[13:-16]

    #relevant columns
    relevant_columns = [
        'activeSessions_site_1', 'activeSessions_site_2',
        'utilization_rate_1', 'utilization_rate_2',
        'total_drawn_power_1', 'total_drawn_power_2',
        'idle_rate_1', 'idle_rate_2'
    ]

    #resample and calculate min, max, mean
    aggregated_data = (
        hourly_data
        .set_index('hour')
        .resample('24h')[relevant_columns]
        .agg(['min', 'max', 'mean'])
    )

    #build new column names
    aggregated_data.columns = [
        f"{col}_{stat}" for col, stat in aggregated_data.columns
    ]

    # Step 6: Reset index and rename time column to 'day'
    aggregated_data = aggregated_data.reset_index().rename(columns={'hour': 'day'})

    aggregated_data.to_csv(KPI_dataset_path, index=False)
    print("aggregated KPI dataset saved to ", KPI_dataset_path)

    return aggregated_data

if __name__ == "__main__":
    add_avg_min_max_KPIs(nn_dataset_path="data/processed/hourly_avg_power.csv", KPI_dataset_path='data/processed/daily_avg_min_max_KPIs.csv')