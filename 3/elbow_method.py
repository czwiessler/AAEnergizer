from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import pandas as pd

file_path = '../data/charging_sessions_cleaned.csv'
data = pd.read_csv(file_path)

print(data.head())
# Auswahl der relevanten Merkmale
features = data[['kWhDelivered', 'duration', 'chargingPower']].copy()

# Normalisierung der Daten
scaler = StandardScaler()
normalized_features = scaler.fit_transform(features)

# Bestimmung der optimalen Anzahl an Clustern mit der Elbow-Methode
inertia = []
cluster_range = range(1, 11)

for k in cluster_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(normalized_features)
    inertia.append(kmeans.inertia_)

# Elbow-Plot erstellen
plt.figure(figsize=(8, 5))
plt.plot(cluster_range, inertia, marker='o', linestyle='--')
plt.title('Elbow-Methode zur Bestimmung der optimalen Clusteranzahl')
plt.xlabel('Anzahl der Cluster (k)')
plt.ylabel('Inertia')
plt.grid(True)
plt.show()
