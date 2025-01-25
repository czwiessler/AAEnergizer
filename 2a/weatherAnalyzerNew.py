import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv("data/raw/weather_burbank_airport.csv")

# Convert timestamp to datetime and extract month
data['timestamp'] = pd.to_datetime(data['timestamp'])
data['month'] = data['timestamp'].dt.month

# Define seasons
season_mapping = {
    12: 'Winter', 1: 'Winter', 2: 'Winter',
    3: 'Spring', 4: 'Spring', 5: 'Spring',
    6: 'Summer', 7: 'Summer', 8: 'Summer',
    9: 'Fall', 10: 'Fall', 11: 'Fall'
}
data['season'] = data['month'].map(season_mapping)

# Group by season and calculate mean precipitation and felt_temperature
seasonal_averages = data.groupby('season').agg({
    'precipitation': 'mean',
    'felt_temperature': 'mean'
}).reindex(['Winter', 'Spring', 'Summer', 'Fall'])  # Ensure proper seasonal order


# Plot seasonal precipitation averages
plt.figure(figsize=(10, 5))
plt.bar(seasonal_averages.index, seasonal_averages['precipitation'], color='skyblue')
plt.title('Seasonal Precipitation Averages')
plt.xlabel('Season')
plt.ylabel('Average Precipitation (mm)')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# Plot seasonal felt_temperature averages
plt.figure(figsize=(10, 5))
plt.bar(seasonal_averages.index, seasonal_averages['felt_temperature'], color='salmon')
plt.title('Seasonal felt_temperature Averages')
plt.xlabel('Season')
plt.ylabel('Average felt_temperature (°C)')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
