import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

df = pd.read_csv("../data/charging_sessions_cleaned.csv")
#Filter important values
df_cleaned = df[(df['duration'] <= 100) & (df['kWhDelivered'] <= 100)]
#Select columns for clustering
data = df_cleaned[['duration', 'kWhDelivered']]

#Scale data for clustering
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# Apply KMeans clustering with 3 clusters
kmeans = KMeans(n_clusters=3, random_state=42)
df_cleaned['cluster'] = kmeans.fit_predict(data_scaled)

# Save the KMeans model and scaled data
joblib.dump(kmeans, "../3/kmeans_model.pkl")
np.save("../3/data_scaled.npy", data_scaled)

cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)

print("Cluster-Zentren (3 Cluster):")
for i, center in enumerate(cluster_centers):
    print(f"Cluster {i+1}: Ladezeit = {center[0]:.2f} Stunden, Energie = {center[1]:.2f} kWh")


#Plot the cluster
plt.figure(figsize=(10, 6))
plt.scatter(data['duration'], data['kWhDelivered'], c=df_cleaned['cluster'], cmap='viridis', s=50, alpha=0.6)
plt.xlabel("Ladezeit (Stunden)")
plt.ylabel("Gelieferte Energie (kWh)")
plt.title("Clusteranalyse mit 3 Clustern")
plt.colorbar(label="Cluster")
plt.grid(True)
plt.show()

#Save the data with cluster labels
df_cleaned.to_csv("../data/charging_sessions_with_clusters.csv", index=False)
print("Die Cluster wurden zugewiesen und in einer neuen CSV-Datei gespeichert.")