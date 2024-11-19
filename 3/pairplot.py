import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# load the data
file_path = '3/clustered_charging_data.csv'
charging_data = pd.read_csv(file_path)
# Auswahl weniger Dimensionen für Pairplot
subset = charging_data[[
    "kWhDelivered", "connection_duration", "charging_duration",
    "post_charging_duration", "cluster"
]]

# Pairplot
sns.pairplot(subset, hue="cluster", palette="tab10", diag_kind="kde", height=2.5)
plt.show()
