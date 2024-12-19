import joblib
import numpy as np
from sklearn.metrics import silhouette_score

# Load the pre-trained KMeans model and the scaled data
kmeans = joblib.load("../3/kmeans_model.pkl")
data_scaled = np.load("../3/data_scaled.npy")

# Calculate the silhouette coefficient to evaluate the clustering quality
silhouette_avg = silhouette_score(data_scaled, kmeans.labels_)


print(f"Silhouetten-Koeffizient: {silhouette_avg}")