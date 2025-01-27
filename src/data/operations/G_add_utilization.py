import pandas as pd

def add_utilization(dataset_path):
    # Lade den Datensatz
    dataFrame = pd.read_csv(dataset_path, parse_dates=["connectionTime", "disconnectTime", "doneChargingTime"])

    # Extrahiere Stunde und Tag für die Aggregation
    dataFrame['hour'] = dataFrame['connectionTime'].dt.floor('h')  # Rundet auf die nächste volle Stunde

    # Berechne die stündliche Anzahl aktiver Sessions
    hourly_utilization = (
        dataFrame.groupby(['hour', 'siteID'])
        .agg(active_sessions=('sessionID', 'count'))  # Zähle die Anzahl der Sessions
        .reset_index()
    )

    # Merge der neuen Daten zurück in das Original-DataFrame
    dataFrame = pd.merge(dataFrame, hourly_utilization, how='left', on=['hour', 'siteID'])

    # Speichere das aktualisierte DataFrame zurück in die CSV
    dataFrame.to_csv(dataset_path, index=False)


if __name__ == "__main__":
    add_utilization(dataset_path="data/processed/charging_sessions_cleaned.csv")