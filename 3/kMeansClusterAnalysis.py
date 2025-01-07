import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pytz import timezone
from datetime import datetime


df = pd.read_csv("../data/charging_sessions_cleaned.csv")
#Filter important values
df_cleaned = df[(df['duration'] <= 100) & (df['kWhDelivered'] <= 100)]
#Select columns for clustering
data = df_cleaned[['duration', 'kWhDelivered']]

# 1. CSV-Daten laden
file_path = "../data/processed/charging_sessions_cleaned.csv"
df = pd.read_csv(file_path, parse_dates=["connectionTime", "disconnectTime"])


# 2. Zeitzonenanpassung basierend auf der Spalte 'timezone'
def convert_to_local_timezone(row):
    if pd.isnull(row['timezone']):
        return row['connectionTime'], row['disconnectTime']
    local_tz = timezone(row['timezone'])
    if row['connectionTime'].tzinfo is None:
        connection_time = row['connectionTime'].tz_localize('UTC').tz_convert(local_tz)
    else:
        connection_time = row['connectionTime'].tz_convert(local_tz)
    if row['disconnectTime'].tzinfo is None:
        disconnect_time = row['disconnectTime'].tz_localize('UTC').tz_convert(local_tz)
    else:
        disconnect_time = row['disconnectTime'].tz_convert(local_tz)
    return connection_time, disconnect_time

# Lokale Zeitzonenanpassung durchführen
converted_times = df.apply(lambda row: convert_to_local_timezone(row), axis=1)
df['connectionTime'], df['disconnectTime'] = zip(*converted_times)

# 3. Feature-Engineering
df['duration'] = (df['disconnectTime'] - df['connectionTime']).dt.total_seconds() / 3600
df['hourOfDay'] = df['connectionTime'].dt.hour
features = df[['kWhDelivered', 'duration', 'hourOfDay', 'siteID']].dropna()

# 4. Daten normalisieren
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# 5. Elbow-Methode und Silhouetten-Score zur Bestimmung der optimalen Anzahl von Clustern
inertia = []
silhouette_scores = []
cluster_range = range(2, 11)

for k in cluster_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_features)
    inertia.append(kmeans.inertia_)
    sil_score = silhouette_score(scaled_features, kmeans.labels_)
    silhouette_scores.append(sil_score)

# Elbow-Plot erstellen
plt.figure(figsize=(8, 6))
plt.plot(cluster_range, inertia, marker='o', linestyle='--')
plt.title("Elbow-Methode zur Bestimmung der optimalen Clusteranzahl")
plt.xlabel("Anzahl der Cluster (k)")
plt.ylabel("Inertia (Summe der quadrierten Abstände)")
plt.grid(True)
plt.show()

# Silhouetten-Score-Plot erstellen
plt.figure(figsize=(8, 6))
plt.plot(cluster_range, silhouette_scores, marker='o', linestyle='--', color='orange')
plt.title("Silhouetten-Score zur Bestimmung der optimalen Clusteranzahl")
plt.xlabel("Anzahl der Cluster (k)")
plt.ylabel("Silhouetten-Score")
plt.grid(True)
plt.show()

# Benutzer kann die optimale Anzahl der Cluster auswählen
optimal_k = int(input("Gib die optimale Anzahl der Cluster (k) ein: "))

# 6. Clusteranalyse mit der optimalen Anzahl von Clustern
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
features['cluster'] = kmeans.fit_predict(scaled_features)

# Cluster-Zentren zurückskalieren
cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)

# Cluster-Zentren ausgeben
print(f"\nCluster-Zentren ({optimal_k} Cluster):")
for i, center in enumerate(cluster_centers):
    print(f"Cluster {i+1}:")
    print(f"  Gelieferte Energie = {center[0]:.2f} kWh")
    print(f"  Ladezeit = {center[1]:.2f} Stunden")
    print(f"  Stunde des Tages = {center[2]:.2f}")
    print(f"  Standort-ID = {center[3]:.2f}")
    print()

# 7. Grafische Darstellung der Clusteranalyse
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

scatter = ax.scatter(
    features['kWhDelivered'],
    features['duration'],
    features['hourOfDay'],
    c=features['cluster'], cmap='viridis', s=50, alpha=0.6
)

# Achsen beschriften
ax.set_xlabel("Gelieferte Energie (kWh)")
ax.set_ylabel("Ladezeit (Stunden)")
ax.set_zlabel("Stunde des Tages")
ax.set_title(f"Clusteranalyse mit {optimal_k} Clustern")

# Farbskala hinzufügen
cbar = plt.colorbar(scatter, pad=0.1, ax=ax)
cbar.set_label("Cluster")

plt.show()
