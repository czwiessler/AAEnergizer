import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np

# CSV-Daten laden
file_path = "../data/processed/charging_sessions_cleaned.csv"
df = pd.read_csv(file_path, parse_dates=["connectionTime", "disconnectTime"])

# Feature-Engineering
df['duration'] = (df['disconnectTime'] - df['connectionTime']).dt.total_seconds() / 3600  # Dauer in Stunden
df['hourOfDay'] = df['connectionTime'].dt.hour

# Zyklische Transformation der Stunde (sin und cos)
df['hour_sin'] = np.sin(2 * np.pi * df['hourOfDay'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hourOfDay'] / 24)

# Daten für Clustering: Dauer und zyklische Transformation der Stunde
features = ['duration', 'hour_sin', 'hour_cos']

# Filtere NaN-Werte
df_filtered = df[features + ['hourOfDay']].dropna()  # Hier fügen wir 'hourOfDay' hinzu

# Skalierung der Daten
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df_filtered[['duration', 'hour_sin', 'hour_cos']])

# KMeans-Clustering mit 4 Clustern
n_clusters = 4
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
df_filtered['cluster'] = kmeans.fit_predict(scaled_features)

# 2D-Plot der Cluster-Ergebnisse (Stunde des Tages vs. Dauer)
plt.figure(figsize=(10, 6))

# Plot für hourOfDay auf der x-Achse und Dauer auf der y-Achse
# Hier verwenden wir 'hourOfDay', aber die Punkte werden zyklisch angezeigt
plt.scatter(df_filtered['hourOfDay'], df_filtered['duration'], c=df_filtered['cluster'], cmap='viridis', alpha=0.6)

# Hinzufügen der Labels
plt.xlabel('Hour of Day')
plt.ylabel('Duration (hours)')
plt.title(f'Cluster-Diagramm für Stunde des Tages und Dauer mit {n_clusters} Clustern')

# Hinzufügen der Farblegende
plt.colorbar(label='Cluster')

# Anzeigen
plt.show()

# Berechnung des Circular Mean für 'hourOfDay' innerhalb jedes Clusters
def calculate_circular_mean(hour_sin, hour_cos):
    mean_sin = np.mean(hour_sin)
    mean_cos = np.mean(hour_cos)
    mean_angle = np.arctan2(mean_sin, mean_cos)
    mean_hour = (mean_angle * 24 / (2 * np.pi)) % 24
    return mean_hour

# Cluster-Statistiken mit circular mean
df_filtered['hour_circular_mean'] = df_filtered.groupby('cluster').apply(
    lambda group: calculate_circular_mean(group['hour_sin'], group['hour_cos'])
).reset_index(level=0, drop=True)

# Ausgabe der Cluster-Statistiken
cluster_stats = df_filtered.groupby('cluster').agg({
    'duration': ['mean', 'std'],
    'hour_circular_mean': ['mean']
}).round(2)

print(cluster_stats)
