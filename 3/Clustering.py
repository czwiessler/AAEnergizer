import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from datetime import datetime
import ast

# Datei laden
file_path = "data/charging_sessions.csv"  # Pfad zur CSV-Datei
charging_data = pd.read_csv(file_path)

# Hilfsfunktion zur Berechnung von Dauer in Minuten
def safe_calculate_duration(start, end):
    try:
        start = datetime.fromisoformat(str(start).replace("Z", "+00:00"))
        end = datetime.fromisoformat(str(end).replace("Z", "+00:00"))
        return (end - start).total_seconds() / 60
    except Exception:
        return np.nan

# Daten vorbereiten: Dauer berechnen
charging_data["connection_duration"] = charging_data.apply(
    lambda row: safe_calculate_duration(row["connectionTime"], row["disconnectTime"]), axis=1
)
charging_data["charging_duration"] = charging_data.apply(
    lambda row: safe_calculate_duration(row["connectionTime"], row["doneChargingTime"]), axis=1
)
charging_data["post_charging_duration"] = charging_data["connection_duration"] - charging_data["charging_duration"]

# Felder aus userInputs extrahieren
charging_data["userInputs_parsed"] = charging_data["userInputs"].apply(
    lambda x: ast.literal_eval(x)[0] if isinstance(x, str) else {}
)
charging_data["kWhRequested"] = charging_data["userInputs_parsed"].apply(lambda x: x.get("kWhRequested", 0))
charging_data["milesRequested"] = charging_data["userInputs_parsed"].apply(lambda x: x.get("milesRequested", 0))
charging_data["minutesAvailable"] = charging_data["userInputs_parsed"].apply(lambda x: x.get("minutesAvailable", 0))
charging_data["paymentRequired"] = charging_data["userInputs_parsed"].apply(
    lambda x: 1 if x.get("paymentRequired", False) else 0
)

# Relevante Features auswählen
features = charging_data[[
    "kWhDelivered", "connection_duration", "charging_duration",
    "post_charging_duration", "kWhRequested", "milesRequested",
    "minutesAvailable", "paymentRequired"
]].fillna(0)

# Features skalieren
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# K-Means Clustering durchführen
kmeans = KMeans(n_clusters=4, random_state=42)
charging_data["cluster"] = kmeans.fit_predict(scaled_features)

# Silhouette-Score berechnen
silhouette_avg = silhouette_score(scaled_features, charging_data["cluster"])
print(f"Silhouette Score: {silhouette_avg}")

# Ergebnisse speichern oder anzeigen
charging_data_clustered = charging_data[[
    "id", "kWhDelivered", "connection_duration",
    "charging_duration", "post_charging_duration",
    "kWhRequested", "milesRequested", "minutesAvailable",
    "paymentRequired", "cluster"
]]

# Ergebnisse als CSV speichern
output_path = "3/clustered_charging_data.csv"
charging_data_clustered.to_csv(output_path, index=False)
print(f"Clustered data saved to {output_path}")
