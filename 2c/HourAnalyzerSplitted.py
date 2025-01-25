import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Read the data
data = pd.read_csv('../data/raw/charging_sessions.csv')

# Parse datetime columns (they are in UTC)
data['connectionTime'] = pd.to_datetime(data['connectionTime'], utc=True)
data['disconnectTime'] = pd.to_datetime(data['disconnectTime'], utc=True)
data['doneChargingTime'] = pd.to_datetime(data['doneChargingTime'], utc=True)

# Calculate session duration in minutes
data['session_duration_minutes'] = (data['disconnectTime'] - data['connectionTime']).dt.total_seconds() / 60.0

# List of siteIDs
site_ids = data['siteID'].unique()

# Create a figure for the plots
plt.figure(figsize=(15, 15))

# Define colors for each site
colors = {site_id: color for site_id, color in zip(site_ids, ['blue', 'orange', 'green', 'red', 'purple', 'brown'])}

# Initialize a dictionary to store KPI DataFrames for each site
site_kpis = {}

# Loop over each siteID
for site_id in site_ids:
    # Filter data for the site
    site_data = data[data['siteID'] == site_id].copy()

    # Total number of stations at this site
    total_stations = site_data['stationID'].nunique()

    # Create an empty DataFrame to hold per-hour occupancy data
    occupancy_data = []

    # Iterate over each session to calculate occupancy per hour
    for idx, row in site_data.iterrows():
        station_id = row['stationID']
        connect_time = row['connectionTime']
        disconnect_time = row['disconnectTime']

        # Generate hourly periods overlapping the session in UTC
        session_hours = pd.date_range(
            start=connect_time.floor('h'),
            end=disconnect_time.ceil('h'),
            freq='h',
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

    # Convert hour to local time and extract hour of day
    occupancy_df['hour_local'] = occupancy_df['hour'].dt.tz_convert('America/Los_Angeles')
    occupancy_df['hour_of_day'] = occupancy_df['hour_local'].dt.hour

    # Calculate total occupancy minutes per hour of day
    hourly_occupancy = occupancy_df.groupby('hour_of_day')['occupancy_minutes'].sum().reset_index()

    # Calculate total possible occupancy minutes per hour across all days
    # Get the number of days in the dataset for this site
    date_range = pd.date_range(
        start=site_data['connectionTime'].dt.date.min(),
        end=site_data['disconnectTime'].dt.date.max(),
        freq='D'
    )
    num_days = len(date_range)

    hourly_occupancy['total_possible_minutes'] = num_days * total_stations * 60  # 60 minutes per hour

    # Calculate utilization rate
    hourly_occupancy['utilization_rate'] = (hourly_occupancy['occupancy_minutes'] / hourly_occupancy[
        'total_possible_minutes']) * 100

    # For other KPIs, assign sessions to hours based on connectionTime in local time
    site_data['connectionTime_local'] = site_data['connectionTime'].dt.tz_convert('America/Los_Angeles')
    site_data['hour_of_day'] = site_data['connectionTime_local'].dt.hour

    # Calculate average kWh per session per hour of day
    hourly_kwh = site_data.groupby('hour_of_day')['kWhDelivered'].mean().reset_index()
    hourly_kwh.rename(columns={'kWhDelivered': 'average_kWh_per_session'}, inplace=True)

    # Calculate average session duration per hour of day
    hourly_duration = site_data.groupby('hour_of_day')['session_duration_minutes'].mean().reset_index()
    hourly_duration.rename(columns={'session_duration_minutes': 'average_session_duration_minutes'}, inplace=True)

    # Merge all KPIs into a single DataFrame
    hourly_kpis = pd.merge(
        hourly_occupancy[['hour_of_day', 'utilization_rate']],
        hourly_kwh,
        on='hour_of_day',
        how='left'
    )
    hourly_kpis = pd.merge(
        hourly_kpis,
        hourly_duration,
        on='hour_of_day',
        how='left'
    )

    # Store the DataFrame for later use
    site_kpis[site_id] = hourly_kpis

    # Sort by hour of day
    hourly_kpis.sort_values('hour_of_day', inplace=True)

    # Plot the KPIs over the hours of the day
    plt.subplot(3, 1, 1)
    plt.plot(hourly_kpis['hour_of_day'], hourly_kpis['utilization_rate'], marker='o', color=colors[site_id],
             label=f'Site {site_id}')
    plt.title('Average Utilization Rate by Hour of Day')
    plt.xlabel('Hour of Day')
    plt.ylabel('Utilization Rate (%)')
    plt.xticks(range(0, 24))
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(hourly_kpis['hour_of_day'], hourly_kpis['average_kWh_per_session'], marker='o', color=colors[site_id],
             label=f'Site {site_id}')
    plt.title('Average kWh per Session by Hour of Day')
    plt.xlabel('Hour of Day')
    plt.ylabel('Average kWh per Session')
    plt.xticks(range(0, 24))
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(hourly_kpis['hour_of_day'], hourly_kpis['average_session_duration_minutes'], marker='o',
             color=colors[site_id], label=f'Site {site_id}')
    plt.title('Average Session Duration by Hour of Day')
    plt.xlabel('Hour of Day')
    plt.ylabel('Average Session Duration (minutes)')
    plt.xticks(range(0, 24))
    plt.legend()

plt.tight_layout()
plt.show()
