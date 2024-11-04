import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sb

# Load and preprocess data
dataFrame = pd.read_csv("../data/charging_sessions.csv", parse_dates=["connectionTime", "disconnectTime", "doneChargingTime"])
dataFrame["duration"] = (dataFrame["disconnectTime"] - dataFrame["connectionTime"]).dt.total_seconds() / 60  # Convert duration to minutes

# Split by site
dfSiteOne = dataFrame.loc[dataFrame["siteID"] == 1].copy()  # Use .copy() to avoid SettingWithCopyWarning
dfSiteTwo = dataFrame.loc[dataFrame["siteID"] == 2].copy()  # Use .copy() to avoid SettingWithCopyWarning

# Add day of the week for the first plot
dfSiteOne["dayOfWeek"] = dfSiteOne["connectionTime"].dt.day_name()
dfSiteTwo["dayOfWeek"] = dfSiteTwo["connectionTime"].dt.day_name()

# Calculate mean kWhDelivered for each day of the week
site_one_kWh_means = dfSiteOne.groupby("dayOfWeek")["kWhDelivered"].mean()
site_two_kWh_means = dfSiteTwo.groupby("dayOfWeek")["kWhDelivered"].mean()

# Ensure days are ordered from Monday to Sunday
ordered_days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
site_one_kWh_means = site_one_kWh_means.reindex(ordered_days)
site_two_kWh_means = site_two_kWh_means.reindex(ordered_days)

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

# Set up subplots
fig, axes = plt.subplots(2, 1, figsize=(10, 8))

# Plot Mean kWh Delivered for each day of the week
sb.lineplot(data=site_one_kWh_means, label="Site One", marker="o", ax=axes[0])
sb.lineplot(data=site_two_kWh_means, label="Site Two", marker="o", ax=axes[0])
axes[0].set_xlabel("Day of the Week")
axes[0].set_ylabel("Mean kWh Delivered")
axes[0].set_title("Mean kWh Delivered for Each Day of the Week")
axes[0].legend()
axes[0].tick_params(axis='x', rotation=45)

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

plt.tight_layout()
plt.show()
