import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sb

##################################diagram 1##################################
# Load and preprocess data
df_diagram_one = pd.read_csv("../data/processed/charging_sessions_cleaned.csv", parse_dates=["connectionTime", "disconnectTime", "doneChargingTime"])

# Split by site
dfSiteOne = df_diagram_one.loc[df_diagram_one["siteID"] == 1].copy()  # Use .copy() to avoid SettingWithCopyWarning
dfSiteTwo = df_diagram_one.loc[df_diagram_one["siteID"] == 2].copy()  # Use .copy() to avoid SettingWithCopyWarning


# Set up duration bins in 15-minute intervals
# duration_bins = list(range(0, int(dataFrame["duration"].max() // 15 + 1) * 15 + 1, 15))  # e.g., 0, 15, 30, ..., max duration in dataset
duration_bins = list(range(0, 1020, 15)) #1020 weil ausreißer (duration größer 17h) interessieren nich

# Bin the durations and calculate the count of sessions in each duration bin
dfSiteOne["duration_bin"] = pd.cut(dfSiteOne["duration"], bins=duration_bins, right=False)
dfSiteTwo["duration_bin"] = pd.cut(dfSiteTwo["duration"], bins=duration_bins, right=False)

# Calculate the number of sessions in each duration bin
site_one_duration_dist = dfSiteOne["duration_bin"].value_counts().sort_index()
site_two_duration_dist = dfSiteTwo["duration_bin"].value_counts().sort_index()

# Convert index to numeric for seaborn compatibility and adjust to hours for x-axis labeling
site_one_duration_dist.index = site_one_duration_dist.index.categories.mid / 60  # Convert from minutes to hours
site_two_duration_dist.index = site_two_duration_dist.index.categories.mid / 60  # Convert from minutes to hours
############################################################################



##################################diagram 2##################################
# Load and preprocess data
df_diagram_2 = pd.read_csv("../data/processed/charging_sessions_cleaned.csv", parse_dates=["connectionTime", "disconnectTime", "doneChargingTime"])

# Ensure datetime columns are parsed
df_diagram_2['connectionTime'] = pd.to_datetime(df_diagram_2['connectionTime'])

# Extract the day of the week from the connectionTime
df_diagram_2['dayOfWeek'] = df_diagram_2['connectionTime'].dt.day_name()

# Count sessions per siteID and day of the week
site_day_counts = df_diagram_2.groupby(['siteID', 'dayOfWeek']).size().reset_index(name='sessionCount')

# Ensure the days of the week are ordered correctly
order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
site_day_counts['dayOfWeek'] = pd.Categorical(site_day_counts['dayOfWeek'], categories=order, ordered=True)

# Sort data by day of the week for proper plotting
site_day_counts = site_day_counts.sort_values(by='dayOfWeek')
############################################################################




# Set up subplots
fig, axes = plt.subplots(2, 1, figsize=(10, 8))

# Plot Duration Distribution in 15-Minute Bins with Hours on x-axis
sb.lineplot(x=site_one_duration_dist.index, y=site_one_duration_dist.values, label="Site One", marker="o", ax=axes[1])
sb.lineplot(x=site_two_duration_dist.index, y=site_two_duration_dist.values, label="Site Two", marker="o", ax=axes[1])
axes[1].set_xlabel("Duration (Hours)")
axes[1].set_ylabel("Number of Sessions")
axes[1].set_title("Charging Duration Distribution")
axes[1].legend()

# Adjust x-axis ticks to show hours in 0.25-hour increments
axes[1].set_xticks([i * 0.5 for i in range(int(1020 / 60 / 0.5) + 1)])  # Converts 1500 mins to hours with 0.25h steps
axes[1].set_xlim(0, 1020 / 60)  # Limit x-axis to 1500 minutes, converted to hours


# Plot: Number of sessions per day of the week for each siteID
for site in site_day_counts['siteID'].unique():
    site_data = site_day_counts[site_day_counts['siteID'] == site]
    plt.plot(site_data['dayOfWeek'], site_data['sessionCount'], marker='o', label=f'Site {site}')

plt.title("Number of Sessions per Day of the Week for Each Site")
plt.xlabel("Day of the Week")
plt.ylabel("Number of Sessions")
plt.xticks(rotation=45)
plt.legend(title="Site ID")


plt.tight_layout()
plt.show()
