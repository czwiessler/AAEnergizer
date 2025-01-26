import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import ast

# CSV-Daten laden
file_path = "../data/processed/charging_sessions_cleaned.csv"
df = pd.read_csv(file_path, parse_dates=["connectionTime", "disconnectTime"])

# Feature-Engineering
df['duration'] = (df['disconnectTime'] - df['connectionTime']).dt.total_seconds() / 3600
df['hourOfDay'] = df['connectionTime'].dt.hour

def extract_wh_per_mile(user_inputs):
    try:
        if isinstance(user_inputs, str):
            user_inputs = ast.literal_eval(user_inputs)
        if isinstance(user_inputs, list) and len(user_inputs) > 0:
            return user_inputs[0].get('WhPerMile', None)
    except Exception as e:
        print(f"Fehler beim Extrahieren von WhPerMile: {e}")
    return None

df['WhPerMile'] = df['userInputs'].apply(lambda x: extract_wh_per_mile(x))

# Definition der Clustering-Konfiguration
clustering_configurations = [
    (['kWhDelivered', 'chargingPower'], 3),
    (['chargingPower', 'kWhDelivered'], 3),
    (['hourOfDay', 'chargingPower'], 3),
    (['chargingPower', 'hourOfDay'], 3)

]

# Daten normalisieren und Clustering durchführen
scaler = StandardScaler()

for features, n_clusters in clustering_configurations:
    selected_data = df[features].dropna()
    scaled_features = scaler.fit_transform(selected_data)

    # KMeans-Clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    selected_data['cluster'] = kmeans.fit_predict(scaled_features)

    # Plot erstellen
    if len(features) == 2:
        plt.figure(figsize=(8, 6))
        plt.scatter(
            selected_data[features[0]],
            selected_data[features[1]],
            c=selected_data['cluster'],
            cmap='viridis',
            alpha=0.6
        )
        plt.xlabel(features[0])
        plt.ylabel(features[1])
        plt.title(f'Cluster-Diagramm für {features} mit {n_clusters} Clustern')
        plt.colorbar(label='Cluster')
        plt.show()

    elif len(features) == 3:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(
            selected_data[features[0]],
            selected_data[features[1]],
            selected_data[features[2]],
            c=selected_data['cluster'],
            cmap='viridis',
            alpha=0.6
        )
        ax.set_xlabel(features[0])
        ax.set_ylabel(features[1])
        ax.set_zlabel(features[2])
        ax.set_title(f'3D-Cluster-Diagramm für {features} mit {n_clusters} Clustern')
        plt.colorbar(scatter, label='Cluster')
        plt.show()
