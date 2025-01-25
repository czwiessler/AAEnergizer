import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sb

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


































