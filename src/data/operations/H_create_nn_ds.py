import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
import numpy as np

def add_holidays(df):
    df['hour'] = pd.to_datetime(df['hour'])
    df['date'] = df['hour'].dt.date

    cal = USFederalHolidayCalendar()
    holidays = cal.holidays(start=df['hour'].min(), end=df['hour'].max()).to_pydatetime()
    holiday_dates = set([d.date() for d in holidays])

    df['is_holiday'] = df['date'].apply(lambda x: 1 if x in holiday_dates else 0)
    df['is_weekend'] = df['hour'].dt.weekday.apply(lambda x: 1 if x >= 5 else 0)
    df['is_vacation'] = df['hour'].dt.month.apply(lambda x: 1 if x in [6, 7, 8, 12] else 0)

    # verschiebe die zeilen holiday weekend und vac 24 stunden nach oben, da wir zukünftige werte vorhersagen wollen
    df['is_holiday'] = df['is_holiday'].shift(-24)
    df['is_weekend'] = df['is_weekend'].shift(-24)
    df['is_vacation'] = df['is_vacation'].shift(-24)


    def season(month):
        if month in [12, 1, 2]:
            return 'Winter'
        elif month in [3, 4, 5]:
            return 'Spring'
        elif month in [6, 7, 8]:
            return 'Summer'
        else:
            return 'Fall'

    df['season'] = df['hour'].dt.month.apply(season)
    df.drop(columns=['date'], inplace=True)

    return df

def create_nn_ds(dataset_path, weather_dataset_path, nn_dataset_path):
    df = pd.read_csv(dataset_path)
    weather_df = pd.read_csv(weather_dataset_path)

    df['connectionTime'] = pd.to_datetime(df['connectionTime'])
    df['disconnectTime'] = pd.to_datetime(df['disconnectTime'])
    df['hourly_timestamp'] = df['connectionTime'].dt.floor('h')
    weather_df['timestamp'] = pd.to_datetime(weather_df['timestamp']) - pd.Timedelta(hours=8)
    weather_df['hourly_timestamp'] = weather_df['timestamp'].dt.floor('h')

    df['hourly_timestamp'] = df['hourly_timestamp'].dt.tz_localize(None)
    weather_df['hourly_timestamp'] = weather_df['hourly_timestamp'].dt.tz_localize(None)

    weather_df = weather_df.groupby('hourly_timestamp').agg({
        'temperature': 'mean',
        'precipitation': 'mean'
    }).interpolate(method='linear').reset_index()

    time_intervals = pd.date_range(
        start=df['connectionTime'].min().floor('h'),
        end=df['disconnectTime'].max().ceil('h'),
        freq='h'
    )

    site_ids = df['siteID'].unique()
    hourly_data = {site_id: [] for site_id in site_ids}
    print("Creating hourly data structure...")
    for site_id in site_ids:
        hourly_data[site_id] = [
            {
                'hour': start_time,
                f'avgChargingPower_site_{site_id}': 0,
                f'activeSessions_site_{site_id}': 0
            }
            for start_time in time_intervals
        ]

    print("Calculating hourly average power and active sessions...")
    i = 0
    for start_time in time_intervals:
        end_time = start_time + pd.Timedelta(hours=1)
        for site_id in site_ids:
            site_df = df[df['siteID'] == site_id]
            active_sessions = site_df[
                (site_df['connectionTime'] < end_time) & (site_df['disconnectTime'] > start_time)
            ]
            if not active_sessions.empty:
                avg_power = active_sessions['chargingPower'].mean()
                max_active_sessions = active_sessions['active_sessions'].max()

                for entry in hourly_data[site_id]:
                    if entry['hour'] == start_time:
                        entry[f'avgChargingPower_site_{site_id}'] = avg_power
                        entry[f'activeSessions_site_{site_id}'] = max_active_sessions

        i += 1
        if i % 500 == 0:
            print(f"Processed {i}/{len(time_intervals)} hours...")

    hourly_data_dfs = []
    for site_id, data in hourly_data.items():
        site_df = pd.DataFrame(data)
        site_df['hour'] = pd.to_datetime(site_df['hour']).dt.tz_localize(None)
        hourly_data_dfs.append(site_df)

    merged_df = pd.concat(hourly_data_dfs, axis=1).loc[:, ~pd.concat(hourly_data_dfs, axis=1).columns.duplicated()]

    merged_df = pd.merge(merged_df, weather_df, left_on='hour', right_on='hourly_timestamp', how='left')
    merged_df = add_holidays(merged_df)
    merged_df.drop(columns=['hourly_timestamp'], inplace=True)
    merged_df.to_csv(nn_dataset_path, index=False)

    return merged_df
