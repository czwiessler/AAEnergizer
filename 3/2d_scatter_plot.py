import matplotlib.pyplot as plt
import pandas as pd

# load the data
file_path = '3/clustered_charging_data.csv'
charging_data = pd.read_csv(file_path)

# Scatter-Plot für zwei Dimensionen
plt.figure(figsize=(10, 6))
for cluster in charging_data["cluster"].unique():
    cluster_data = charging_data[charging_data["cluster"] == cluster]
    plt.scatter(cluster_data["kWhDelivered"], cluster_data["connection_duration"], label=f"Cluster {cluster}")

plt.title("Cluster Visualization: kWh Delivered vs Connection Duration")
plt.xlabel("kWh Delivered")
plt.ylabel("Connection Duration (minutes)")
plt.legend()
plt.show()
