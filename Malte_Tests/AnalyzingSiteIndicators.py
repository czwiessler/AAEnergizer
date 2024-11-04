import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sb

dataFrame = pd.read_csv("../data/charging_sessions.csv", parse_dates=["connectionTime", "disconnectTime", "doneChargingTime"])

dataFrame["duration"] = dataFrame["disconnectTime"] - dataFrame["connectionTime"]

dfSiteOne = dataFrame[dataFrame["siteID"] == 1]
dfSiteOne[["duration", "kWhDelivered"]].describe()

dfSiteTwo = dataFrame[dataFrame["siteID"] == 2]
dfSiteTwo[["duration", "kWhDelivered"]].describe()

dfSiteOne["dayOfWeek"] = dfSiteOne["connectionTime"].dt.day_name()
dfSiteTwo["dayOfWeek"] = dfSiteTwo["connectionTime"].dt.day_name()

site_one_means = dfSiteOne.groupby("dayOfWeek")["kWhDelivered"].mean()
site_two_means = dfSiteTwo.groupby("dayOfWeek")["kWhDelivered"].mean()

ordered_days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
site_one_means = site_one_means.reindex(ordered_days)
site_two_means = site_two_means.reindex(ordered_days)

plt.figure(figsize=(10, 6))
sb.lineplot(data=site_one_means, label="Site One", marker="o")
sb.lineplot(data=site_two_means, label="Site Two", marker="o")

plt.xlabel("Day of the Week")
plt.ylabel("Mean kWh Delivered")
plt.title("Mean kWh Delivered for each day")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()