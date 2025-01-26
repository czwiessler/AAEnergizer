import numpy as np
import pandas as pd
import plotly.express as px

# Load predictions, actuals, and hour data
predictions = np.load('test_predictions.npy')
actuals = np.load('test_actuals.npy')
hour_for_x_axis = np.load('hour.npy', allow_pickle=True)  # Hours for x-axis
hour_for_x_axis = hour_for_x_axis[240:]

# Define target columns
target_columns = [
    'avgChargingPower_site_1', 'activeSessions_site_1',
    'avgChargingPower_site_2', 'activeSessions_site_2'
]

# Prepare DataFrame for comparison
df_preds = pd.DataFrame(predictions, columns=[f"pred_{col}" for col in target_columns])
df_actuals = pd.DataFrame(actuals, columns=[f"actual_{col}" for col in target_columns])
df_compare = pd.concat([df_preds, df_actuals], axis=1)
df_compare['hour'] = hour_for_x_axis  # Add the hour column for the x-axis

# Save interactive plots as HTML files
for column in target_columns:
    fig = px.line(
        df_compare,
        x='hour',
        y=[f"pred_{column}", f"actual_{column}"],
        title=f"{column} - Predictions vs Actuals",
        labels={"hour": "Time", "value": "Value"},
        template="plotly"
    )
    file_name = f"{column}_interactive_plot.html"
    fig.write_html(file_name)
    print(f"Interactive plot saved: {file_name}")
