import pandas as pd

def add_utilization(dataset_path):
    # Lade den Datensatz
    dataFrame = pd.read_csv(dataset_path, parse_dates=["connectionTime", "disconnectTime", "doneChargingTime"])

    # Extrahiere Stunde und Tag für die Aggregation
    dataFrame['hour'] = dataFrame['connectionTime'].dt.floor('H')  # Rundet auf die nächste volle Stunde

    # Anzahl der Ladepunkte an jedem Standort (maximale Kapazität, angenommen)
    site_capacity = {
        "1": 50,  # Kapazität Standort 1
        "2": 50   # Kapazität Standort 2
    }

    # Berechne die stündliche Anzahl aktiver Sessions
    hourly_utilization = (
        dataFrame.groupby(['hour', 'siteID'])
        .agg(active_sessions=('sessionID', 'count'))  # Zähle die Anzahl der Sessions
        .reset_index()
    )

    # Mapping der Kapazität basierend auf der siteID
    hourly_utilization['capacity'] = hourly_utilization['siteID'].astype(str).map(site_capacity)

    # Berechne die Nutzung als Anteil der aktiven Sessions zur Kapazität
    hourly_utilization['utilization'] = hourly_utilization['active_sessions'] / hourly_utilization['capacity']

    # Füge Lag-Features hinzu (1 Stunde und 24 Stunden Verzögerung)
    hourly_utilization['utilization_lag_1h'] = hourly_utilization.groupby('siteID')['utilization'].shift(1)
    hourly_utilization['utilization_lag_24h'] = hourly_utilization.groupby('siteID')['utilization'].shift(24)

    # Merge der neuen Daten zurück in das Original-DataFrame
    dataFrame = pd.merge(dataFrame, hourly_utilization, how='left', on=['hour', 'siteID'])

    # Fehlende Werte in den Lag-Features mit 0 füllen
    dataFrame['utilization'].fillna(0, inplace=True)
    dataFrame['utilization_lag_1h'].fillna(0, inplace=True)
    dataFrame['utilization_lag_24h'].fillna(0, inplace=True)

    # Speichere das aktualisierte DataFrame zurück in die CSV
    dataFrame.to_csv(dataset_path, index=False)

    print("Utilization und Lag-Features erfolgreich hinzugefügt!")



if __name__ == "__main__":
    add_utilization(dataset_path="data/processed/charging_sessions_cleaned.csv")