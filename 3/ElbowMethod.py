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

#Sum of squared errors
sse = []

for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(data_scaled)
    sse.append(kmeans.inertia_)

#Plot SSE for each k
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), sse, marker='o')
plt.title('Elbow-Methode: Optimale Clusteranzahl')
plt.xlabel('Anzahl der Cluster')
plt.ylabel('Summe der quadratischen Fehler (SSE)')
plt.grid(True)
plt.show()