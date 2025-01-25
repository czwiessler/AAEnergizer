import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('data/processed/charging_sessions_cleaned.csv')

# Parse datetime columns because theyre in UTC
data['connectionTime'] = pd.to_datetime(data['connectionTime'], utc=True)
data['doneChargingTime'] = pd.to_datetime(data['doneChargingTime'], utc=True)

#handle missing doneChargingTime by setting it to disconnectTime where NaT
data.loc[data['doneChargingTime'].isna(), 'doneChargingTime'] = data['disconnectTime']

#handle missing durationUntilFullCharge by setting it to duration where NaN
data.loc[(data['durationUntilFullCharge'].isna() | data['durationUntilFullCharge'] == 0 ), 'durationUntilFullCharge'] = data['duration']

# Filter for regular working days (Monday to Friday)
data['is_weekday'] = data['connectionTime'].dt.dayofweek < 5  # Monday to Friday are 0-4
data = data[data['is_weekday']]

# Create an empty DataFrame to hold per-hour power allocation data
power_data = []

# Iterate over each session to allocate charging power per hour
for idx, row in data.iterrows():
    station_id = row['stationID']
    connect_time = row['connectionTime']
    done_charging_time = row['doneChargingTime']
    charging_power = row['kWhDelivered'] / row['durationUntilFullCharge']  # Power in kW

    # Generate hourly periods overlapping the charging session
    session_hours = pd.date_range(
        start=connect_time.floor('h'), # flo
        end=done_charging_time.ceil('h'),
        freq='h',
        tz='UTC'
    )

    for hour_start in session_hours:
        hour_end = hour_start + pd.Timedelta(hours=1)
        # Calculate the overlap between the session and the hour
        overlap_start = max(connect_time, hour_start)
        overlap_end = min(done_charging_time, hour_end)
        charging_minutes = (overlap_end - overlap_start).total_seconds() / 60.0
        if charging_minutes > 0:
            power_data.append({
                'stationID': station_id,
                'hour': hour_start,
                'charging_power': charging_power * (charging_minutes / 60.0)  # Scale by hour proportion
            })

# Create a DataFrame from power_data
power_df = pd.DataFrame(power_data)

# Convert hour to local time and extract hour of day
power_df['hour_local'] = power_df['hour'].dt.tz_convert('America/Los_Angeles')
power_df['hour_of_day'] = power_df['hour_local'].dt.hour

# Calculate total charging power per hour of day
hourly_power = power_df.groupby('hour_of_day')['charging_power'].sum().reset_index()
hourly_power.rename(columns={'charging_power': 'total_charging_power_kW'}, inplace=True)

# Plot total charging power per hour of day
plt.figure(figsize=(10, 6))
plt.plot(hourly_power['hour_of_day'], hourly_power['total_charging_power_kW'], marker='o', color='orange')
plt.title('Total Charging Power by Hour of Day (Weekdays Only)')
plt.xlabel('Hour of Day')
plt.ylabel('Total Charging Power (kW)')
plt.xticks(range(0, 24))
plt.grid(True)
plt.show()
