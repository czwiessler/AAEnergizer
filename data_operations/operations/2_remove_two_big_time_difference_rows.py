import pandas as pd

df = pd.read_csv("../../data/charging_sessions_cleaned.csv")

# filter out ids 5c99728ff9af8b5022123831 and 5e7954b0f9af8b090600ec84: their "doneChargingTime" lies approx. 3600 seconds BEHIND "disconnectTime"
# makes no sense
df[df['id'] != '5c99728ff9af8b5022123831'] \
  .query("id != '5e7954b0f9af8b090600ec84'") \
  .to_csv("../../data/charging_sessions_cleaned.csv", index=False)
