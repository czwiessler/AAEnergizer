import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
data = pd.read_csv("../data/raw/charging_sessions.csv", parse_dates=["connectionTime", "disconnectTime", "doneChargingTime"])

# Extract time-related features for analysis
data['hour'] = data['connectionTime'].dt.hour
data['day_of_week'] = data['connectionTime'].dt.day_name()
data['month'] = data['connectionTime'].dt.month
data['season'] = data['connectionTime'].dt.month % 12 // 3 + 1  # Spring=1, Summer=2, Fall=3, Winter=4

# Create subplots
fig, axes = plt.subplots(3, 1, figsize=(12, 18))

# Hourly charging events
sns.histplot(data['hour'], bins=24, kde=True, ax=axes[0])
axes[0].set_title("Charging Events by Hour of the Day")
axes[0].set_xlabel("Hour of Day")
axes[0].set_ylabel("Number of Charging Events")

# Weekly charging events
sns.countplot(data['day_of_week'], order=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"], ax=axes[1])
axes[1].set_title("Charging Events by Day of the Week")
axes[1].set_xlabel("Day of the Week")
axes[1].set_ylabel("Number of Charging Events")

# Seasonal charging events
season_labels = {1: 'Spring', 2: 'Summer', 3: 'Fall', 4: 'Winter'}
data['season'] = data['season'].map(season_labels)
sns.countplot(data['season'], order=['Spring', 'Summer', 'Fall', 'Winter'], ax=axes[2])
axes[2].set_title("Charging Events by Season")
axes[2].set_xlabel("Season")
axes[2].set_ylabel("Number of Charging Events")

# Adjust layout
plt.tight_layout()
plt.show()

# Save the plot
fig.savefig("output/seasonal_analysis.png")
