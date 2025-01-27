from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import pandas as pd
import ast
from itertools import combinations
from pytz import timezone

file_path = '../data/charging_sessions_cleaned.csv'
data = pd.read_csv(file_path, parse_dates=["connectionTime", "disconnectTime"])
def extract_wh_per_mile(user_inputs):
    try:
        if isinstance(user_inputs, str):
            user_inputs = ast.literal_eval(user_inputs)
        if isinstance(user_inputs, list) and len(user_inputs) > 0:
            return user_inputs[0].get('WhPerMile', None)
    except Exception as e:
        print(f"Fehler beim Extrahieren von WhPerMile: {e}")
    return None

data['WhPerMile'] = data['userInputs'].apply(lambda x: extract_wh_per_mile(x))

# 2. Zeitzonenanpassung
local_timezone = timezone("America/Los_Angeles")

if data['connectionTime'].dt.tz is None:
    data['connectionTime'] = data['connectionTime'].dt.tz_localize('UTC').dt.tz_convert(local_timezone)
else:
    data['connectionTime'] = data['connectionTime'].dt.tz_convert(local_timezone)

if data['disconnectTime'].dt.tz is None:
    data['disconnectTime'] = data['disconnectTime'].dt.tz_localize('UTC').dt.tz_convert(local_timezone)
else:
    data['disconnectTime'] = data['disconnectTime'].dt.tz_convert(local_timezone)

# 3. Feature-Engineering
data['duration'] = (data['disconnectTime'] - data['connectionTime']).dt.total_seconds() / 3600
data['hourOfDay'] = data['connectionTime'].dt.hour

features_list = ['kWhDelivered', 'hourOfDay', 'WhPerMile', 'duration', 'durationUntilFullCharge', 'chargingPower']
best_score = -1
best_combination = None
best_inertia = []
best_sil_scores = []
best_normalized_features = None

for r in range(2, len(features_list) + 1):
    for combo in combinations(features_list, r):
        features = data[list(combo)].copy().dropna()
        scaler = StandardScaler()
        normalized_features = scaler.fit_transform(features)

        inertia = []
        sil_scores = []
        cluster_range = range(2, 11)

        for k in cluster_range:
            kmeans = KMeans(n_clusters=k, random_state=42, init='k-means++')
            kmeans.fit(normalized_features)

            inertia.append(kmeans.inertia_)
            sil_score = silhouette_score(normalized_features, kmeans.labels_)
            sil_scores.append(sil_score)

        print(f"Kombination: {combo}, Silhouetten-Score: {sil_scores}")

        max_sil_score = max(sil_scores)
        if max_sil_score > best_score:
            best_score = max_sil_score
            best_combination = combo
            best_inertia = inertia
            best_sil_scores = sil_scores
            best_normalized_features = normalized_features

print(f"Beste Kombination: {best_combination}, Bester Silhouetten-Score: {best_score}")

plt.figure(figsize=(8, 5))
plt.plot(cluster_range, best_inertia, marker='o', linestyle='--')
plt.title(f'Elbow-Methode für {best_combination}')
plt.xlabel('Anzahl der Cluster (k)')
plt.ylabel('Inertia')
plt.grid(True)
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(cluster_range, best_sil_scores, marker='o', linestyle='--')
plt.title(f'Silhouetten-Score für {best_combination}')
plt.xlabel('Anzahl der Cluster (k)')
plt.ylabel('Silhouetten-Score')
plt.grid(True)
plt.show()
