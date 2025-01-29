import pandas as pd
import numpy as np


def add_sin_cos_time_features(df):
    # Spalte 'hour' in datetime umwandeln
    df['hour'] = pd.to_datetime(df['hour'])

    # Stunde extrahieren
    df['hour_of_day'] = df['hour'].dt.hour

    # Sinus- und Kosinus-Transformation für die Stunden (tageszyklisch)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour_of_day'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour_of_day'] / 24)

    return df


def add_sin_cos_season_features(df):
    # Mapping der Jahreszeiten auf numerische Werte
    season_mapping = {'Spring': 0, 'Summer': 1, 'Fall': 2, 'Winter': 3}
    df['season_num'] = df['season'].map(season_mapping)

    # Sinus- und Kosinus-Transformation für die Jahreszeiten (zyklisch)
    df['season_sin'] = np.sin(2 * np.pi * df['season_num'] / 4)
    df['season_cos'] = np.cos(2 * np.pi * df['season_num'] / 4)

    return df

# Ergänzte Funktion zur Interpolation von Wetterdaten
def interpolate_weather_data(weather_df):
    # Sicherstellen, dass die 'hour'-Spalte als datetime interpretiert wird
    weather_df['hour'] = pd.to_datetime(weather_df['hour'])

    # Sortiere den DataFrame nach Zeitstempel
    weather_df = weather_df.sort_values(by='hour')

    # Lineare Interpolation für die Spalten 'temperature' und 'precipitation'
    weather_df[['temperature', 'precipitation']] = weather_df[['temperature', 'precipitation']].interpolate(
        method='linear', limit_direction='both'
    )

    return weather_df



def prepare_nn_ds(nn_dataset_path):
    # Load the dataset
    df = pd.read_csv(nn_dataset_path)

    # Add sine and cosine transformations for time and season
    df = add_sin_cos_time_features(df)
    df = add_sin_cos_season_features(df)

    # Interpolate weather data
    df = interpolate_weather_data(df)


    # sort the columns Index(['hour', 'avgChargingPower', 'activeSessions', 'temperature',
    #        'precipitation', 'is_holiday', 'is_weekend', 'is_vacation', 'season',
    #        'hour_of_day', 'hour_sin', 'hour_cos', 'season_num', 'season_sin',
    #        'season_cos'],
    #       dtype='object')
    #df = df[['hour', 'temperature', 'precipitation', 'is_holiday', 'is_weekend',
    #         'is_vacation', 'season', 'hour_of_day', 'hour_sin', 'hour_cos',
    #         'season_num', 'season_sin', 'season_cos', 'avgChargingPower_site_1',
    #         'activeSessions_site_1', 'chargingSessions_site_1', 'avgChargingPower_site_2', 'activeSessions_site_2', 'chargingSessions_site_2']]

    # add the column 'day_of_week' to the dataframe, so monday is 0 and sunday is 6
    df['day_of_week'] = df['hour'].dt.dayofweek

    # Save the updated DataFrame back to CSV
    df.to_csv(nn_dataset_path, index=False)

    print("Dataset preparation completed.")

if __name__ == "__main__":
    prepare_nn_ds(nn_dataset_path="data/processed/hourly_avg_power.csv")