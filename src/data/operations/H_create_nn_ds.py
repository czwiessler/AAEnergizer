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
    df['doneChargingTime'] = pd.to_datetime(df['doneChargingTime'])
    weather_df['timestamp'] = pd.to_datetime(weather_df['timestamp']) - pd.Timedelta(hours=8)
    weather_df['hourly_timestamp'] = weather_df['timestamp'].dt.floor('h')

    ############# set disconnectTime where doneChargingTime is null #############
    df.loc[df['doneChargingTime'].isna(), 'doneChargingTime'] = df['disconnectTime']

    weather_df = weather_df.groupby('hourly_timestamp').agg({
        'temperature': 'mean',
        'precipitation': 'mean'
    }).interpolate(method='linear').reset_index()

    # Berechne alle Stunden im Zeitintervall
    time_intervals = pd.date_range(
        start=df['connectionTime'].min().floor('h'),
        end=df['disconnectTime'].max().ceil('h'),
        freq='h'
    )

    site_ids = df['siteID'].unique()
    hourly_data = []

    print("Berechnung stündlicher aktiver Sessions und Ladeleistung...")
    i=0
    for start_time in time_intervals:
        i += 1
        if i % 500 == 0:
            print(f"Processed {i}/{len(time_intervals)} hours...")
        end_time = start_time + pd.Timedelta(hours=1)
        row = {'hour': start_time}

        for site_id in site_ids:
            # Filter für aktive Sessions
            site_df = df[df['siteID'] == site_id]
            active_sessions = site_df[
                (site_df['connectionTime'] < end_time) & (site_df['disconnectTime'] > start_time)
                ]

            charging_sessions = site_df[
                (site_df['connectionTime'] < end_time) & (site_df['doneChargingTime'] > start_time)
                ]

            avg_power = charging_sessions['chargingPower'].mean() if not charging_sessions.empty else 0
            charging_sessions_count = len(charging_sessions)
            active_session_count = len(active_sessions)

            row[f'avgChargingPower_site_{site_id}'] = avg_power
            row[f'chargingSessions_site_{site_id}'] = charging_sessions_count
            row[f'activeSessions_site_{site_id}'] = active_session_count

        hourly_data.append(row)

    hourly_df = pd.DataFrame(hourly_data)
    hourly_df['hour'] = hourly_df['hour'].dt.tz_localize(None)
    weather_df['hourly_timestamp'] = weather_df['hourly_timestamp'].dt.tz_localize(None)

    # Merging Wetterdaten
    hourly_df = pd.merge(hourly_df, weather_df, left_on='hour', right_on='hourly_timestamp', how='left')
    hourly_df.drop(columns=['hourly_timestamp'], inplace=True)

    # jetzt ist jede uhrzeit 2 mal da und es ist immer entweder die eine oder die andere siteID gefüllt, die andere ist null

    # Feiertage hinzufügen (falls implementiert)
    hourly_df = add_holidays(hourly_df)

    # Speichern der Ergebnisse
    hourly_df.to_csv(nn_dataset_path, index=False)
    print("hourly data for prediction and aggregated visualization saved to ", nn_dataset_path)

    return hourly_df
