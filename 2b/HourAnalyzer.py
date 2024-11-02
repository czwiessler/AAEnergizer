import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Read the data
data = pd.read_csv('../data/charging_sessions.csv')

# Parse datetime columns (they are in UTC)
data['connectionTime'] = pd.to_datetime(data['connectionTime'], utc=True)
data['disconnectTime'] = pd.to_datetime(data['disconnectTime'], utc=True)
data['doneChargingTime'] = pd.to_datetime(data['doneChargingTime'], utc=True)

# Calculate session duration in minutes
data['session_duration_minutes'] = (data['disconnectTime'] - data['connectionTime']).dt.total_seconds() / 60.0

# Total number of stations
total_stations = data['stationID'].nunique()

# Create an empty DataFrame to hold per-hour occupancy data
occupancy_data = []

# Iterate over each session to calculate occupancy per hour
for idx, row in data.iterrows():
    station_id = row['stationID']
    connect_time = row['connectionTime']
    disconnect_time = row['disconnectTime']

    # Generate hourly periods overlapping the session in UTC
    session_hours = pd.date_range(
        start=connect_time.floor('H'),
        end=disconnect_time.ceil('H'),
        freq='H',
        tz='UTC'
    )

    for hour_start in session_hours:
        hour_end = hour_start + pd.Timedelta(hours=1)
        # Calculate the overlap between the session and the hour
        overlap_start = max(connect_time, hour_start)
        overlap_end = min(disconnect_time, hour_end)
        occupancy_minutes = (overlap_end - overlap_start).total_seconds() / 60.0
        occupancy_data.append({
            'stationID': station_id,
            'hour': hour_start,
            'occupancy_minutes': occupancy_minutes
        })

# Create a DataFrame from occupancy_data
occupancy_df = pd.DataFrame(occupancy_data)

# Calculate total occupancy minutes per hour
hourly_occupancy = occupancy_df.groupby('hour')['occupancy_minutes'].sum().reset_index()

# Calculate utilization rate
hourly_occupancy['utilization_rate'] = (hourly_occupancy['occupancy_minutes'] / (total_stations * 60)) * 100

# Assign sessions to the starting hour for other KPIs
data['start_hour'] = data['connectionTime'].dt.floor('H')

# Calculate average kWh per session per hour
hourly_kwh = data.groupby('start_hour')['kWhDelivered'].mean().reset_index()
hourly_kwh.rename(columns={'kWhDelivered': 'average_kWh_per_session'}, inplace=True)

# Calculate average session duration per hour
hourly_duration = data.groupby('start_hour')['session_duration_minutes'].mean().reset_index()
hourly_duration.rename(columns={'session_duration_minutes': 'average_session_duration_minutes'}, inplace=True)

# Merge all KPIs into a single DataFrame
hourly_kpis = pd.merge(
    hourly_occupancy[['hour', 'utilization_rate']],
    hourly_kwh,
    left_on='hour',
    right_on='start_hour',
    how='left'
)
hourly_kpis = pd.merge(
    hourly_kpis,
    hourly_duration,
    left_on='hour',
    right_on='start_hour',
    how='left'
)
hourly_kpis = hourly_kpis[['hour', 'utilization_rate', 'average_kWh_per_session', 'average_session_duration_minutes']]

# Convert hours to local timezone for plotting
hourly_kpis['hour_local'] = hourly_kpis['hour'].dt.tz_convert('America/Los_Angeles')

# Sort the DataFrame by local hour for proper plotting
hourly_kpis.sort_values('hour_local', inplace=True)

# Plot the KPIs over time
plt.figure(figsize=(15, 10))

plt.subplot(3, 1, 1)
plt.plot(hourly_kpis['hour_local'], hourly_kpis['utilization_rate'], marker='o')
plt.title('Utilization Rate Over Time')
plt.xlabel('Hour (Local Time)')
plt.ylabel('Utilization Rate (%)')

plt.subplot(3, 1, 2)
plt.plot(hourly_kpis['hour_local'], hourly_kpis['average_kWh_per_session'], marker='o', color='orange')
plt.title('Average kWh per Session Over Time')
plt.xlabel('Hour (Local Time)')
plt.ylabel('Average kWh per Session')

plt.subplot(3, 1, 3)
plt.plot(hourly_kpis['hour_local'], hourly_kpis['average_session_duration_minutes'], marker='o', color='green')
plt.title('Average Session Duration Over Time')
plt.xlabel('Hour (Local Time)')
plt.ylabel('Average Session Duration (minutes)')

plt.tight_layout()
plt.show()
