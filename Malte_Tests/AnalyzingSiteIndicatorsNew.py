import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sb

# Load and preprocess data
dataFrame = pd.read_csv("../data/raw/charging_sessions.csv", parse_dates=["connectionTime", "disconnectTime", "doneChargingTime"])
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
duration_bins = list(range(0, int(dataFrame["duration"].max() // 15 + 1) * 15 + 1, 15))  # e.g., 0, 15, 30, ..., max duration in dataset

# Bin the durations and calculate the distribution for each site
dfSiteOne["duration_bin"] = pd.cut(dfSiteOne["duration"], bins=duration_bins, right=False)
dfSiteTwo["duration_bin"] = pd.cut(dfSiteTwo["duration"], bins=duration_bins, right=False)

# Calculate percentage of sessions in each duration bin
site_one_duration_dist = dfSiteOne["duration_bin"].value_counts(normalize=True).sort_index() * 100
site_two_duration_dist = dfSiteTwo["duration_bin"].value_counts(normalize=True).sort_index() * 100

# Convert index to numeric for seaborn compatibility
site_one_duration_dist.index = site_one_duration_dist.index.categories.mid
site_two_duration_dist.index = site_two_duration_dist.index.categories.mid

# Set up subplots
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Plot Mean kWh Delivered for each day of the week
sb.lineplot(data=site_one_kWh_means, label="Site One", marker="o", ax=axes[0])
sb.lineplot(data=site_two_kWh_means, label="Site Two", marker="o", ax=axes[0])
axes[0].set_xlabel("Day of the Week")
axes[0].set_ylabel("Mean kWh Delivered")
axes[0].set_title("Mean kWh Delivered for Each Day of the Week")
axes[0].legend()
axes[0].tick_params(axis='x', rotation=45)

# Plot Duration Distribution in 15-Minute Bins
sb.lineplot(x=site_one_duration_dist.index, y=site_one_duration_dist.values, label="Site One", marker="o", ax=axes[1])
sb.lineplot(x=site_two_duration_dist.index, y=site_two_duration_dist.values, label="Site Two", marker="o", ax=axes[1])
axes[1].set_xlabel("Duration (Minutes)")
axes[1].set_ylabel("Percentage of Sessions")
axes[1].set_title("Charging Duration Distribution (15-Minute Bins)")
axes[1].legend()
axes[1].set_xlim(0,1000)

plt.tight_layout()
plt.show()
