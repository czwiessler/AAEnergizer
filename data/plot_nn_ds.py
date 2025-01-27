import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px

file_path = "data/processed/hourly_avg_power.csv"
data = pd.read_csv(file_path)
# Convert the 'hour' column to a datetime format for better plotting
data['hour'] = pd.to_datetime(data['hour'])

# Define the columns to plot
columns_to_plot = ['avgChargingPower_site_1', 'activeSessions_site_1',
                   'avgChargingPower_site_2', 'activeSessions_site_2']

# Create individual plots for each column against time
for column in columns_to_plot:
    plt.figure(figsize=(10, 5))
    plt.plot(data['hour'], data[column], label=column)
    plt.xlabel('Time')
    plt.ylabel(column)
    plt.title(f'{column} over Time')
    plt.legend()
    plt.grid(True)
    plt.show()

# Create interactive plots using Plotly
for column in columns_to_plot:
    fig = px.line(data, x='hour', y=column, title=f'{column} over Time',
                  labels={'hour': 'Time', column: column})
    fig.update_layout(autosize=True, template="plotly_white")
    fig.show()
