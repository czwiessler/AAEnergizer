import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
file_path = 'data/raw/weather_burbank_airport.csv'
weather_data = pd.read_csv(file_path)

# Parse timestamp column
weather_data['timestamp'] = pd.to_datetime(weather_data['timestamp'])

# Add useful columns for analysis
weather_data['hour'] = weather_data['timestamp'].dt.hour
weather_data['month'] = weather_data['timestamp'].dt.month
weather_data['day_of_week'] = weather_data['timestamp'].dt.dayofweek

# Plot 1: Average Temperature by Hour of the Day
plt.figure(figsize=(10, 6))
sns.lineplot(data=weather_data, x='hour', y='temperature', estimator='mean')
plt.title('Average Temperature by Hour of the Day')
plt.xlabel('Hour')
plt.ylabel('Temperature (°C)')
plt.show()

# Plot 2: Average Temperature by Month
plt.figure(figsize=(10, 6))
sns.lineplot(data=weather_data, x='month', y='temperature', estimator='mean')
plt.title('Average Monthly Temperature')
plt.xlabel('Month')
plt.ylabel('Temperature (°C)')
plt.show()

# Plot 3: Cloud Cover Distribution by Season
weather_data['season'] = weather_data['month'] % 12 // 3 + 1
plt.figure(figsize=(10, 6))
sns.boxplot(data=weather_data, x='season', y='cloud_cover')
plt.title('Cloud Cover by Season')
plt.xlabel('Season (1=Winter, 2=Spring, 3=Summer, 4=Autumn)')
plt.ylabel('Cloud Cover (%)')
plt.show()

# Plot 4: Precipitation Distribution by Hour of the Day
plt.figure(figsize=(10, 6))
sns.boxplot(data=weather_data, x='hour', y='precipitation')
plt.title('Precipitation by Hour of the Day')
plt.xlabel('Hour')
plt.ylabel('Precipitation (mm)')
plt.show()


# do all plots large plot

# Plot 1: Average Temperature by Hour of the Day
plt.figure(figsize=(20, 12))
plt.subplot(2, 2, 1)
sns.lineplot(data=weather_data, x='hour', y='temperature', estimator='mean')
plt.title('Average Temperature by Hour of the Day')
plt.xlabel('Hour')
plt.ylabel('Temperature (°C)')
plt.grid(True)

# Plot 2: Average Temperature by Month
plt.subplot(2, 2, 2)
sns.lineplot(data=weather_data, x='month', y='temperature', estimator='mean')
plt.title('Average Monthly Temperature')
plt.xlabel('Month')
plt.ylabel('Temperature (°C)')
plt.grid(True)

# Plot 3: Cloud Cover Distribution by Season
weather_data['season'] = weather_data['month'] % 12 // 3 + 1
plt.subplot(2, 2, 3)
sns.boxplot(data=weather_data, x='season', y='cloud_cover')
plt.title('Cloud Cover by Season')
plt.xlabel('Season (1=Winter, 2=Spring, 3=Summer, 4=Autumn)')
plt.ylabel('Cloud Cover (%)')
plt.grid(True)

# Plot 4: Precipitation Distribution by Hour of the Day
plt.subplot(2, 2, 4)
sns.boxplot(data=weather_data, x='hour', y='precipitation')
plt.title('Precipitation by Hour of the Day')
plt.xlabel('Hour')
plt.ylabel('Precipitation (mm)')
plt.grid(True)

plt.tight_layout()
plt.show()

# Save the plots as image
plt.savefig('output/weather_analysis_plots.png')