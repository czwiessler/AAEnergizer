import numpy as np
import pandas as pd
import torch
import plotly.express as px
from src.models.lstm_24h import UtilizationPredictor, PowerDataset

# Load dataset
data_path = "data/processed/test_dataset.csv"
data = pd.read_csv(data_path, parse_dates=["hour"])

# Filter for test period
data = data.sort_values("hour")

# Prepare input and target columns
input_columns = [
    'temperature', 'precipitation', 'is_holiday', 'is_weekend', 'is_vacation',
    'hour_sin', 'hour_cos', 'season_sin', 'season_cos'
]

input_columns = [
    'day_of_week', 'hour_of_day'
]
# one hot encoding for hour_of_day and day_of_week
input_columns = [col for col in data.columns if 'hour_of_day_' in col or 'day_of_week_' in col]

target_columns = [
    'avgChargingPower_site_1', 'activeSessions_site_1',
    'avgChargingPower_site_2', 'activeSessions_site_2'
]

forecast_horizon = 1
sequence_length = 24

# Load the model
model_path = "models/24h-model-epoch=00-val_loss=37.55212.ckpt"
model = UtilizationPredictor.load_from_checkpoint(
    checkpoint_path=model_path,
    input_size=len(input_columns),
    hidden_size=128,
    num_layers=2,
    output_size=len(target_columns),
    forecast_horizon=forecast_horizon,
    learning_rate=0.0001,
)
checkpoint = torch.load(model_path, map_location="cuda")
model.load_state_dict(checkpoint["state_dict"])
model.eval()

# Define parameters for prediction
prediction_start_time = pd.Timestamp("2020-01-07 04:00:00")
prediction_end_time = pd.Timestamp("2020-02-29 03:00:00")
prediction_data = data[(data["hour"] >= prediction_start_time) & (data["hour"] < prediction_end_time)]

# Debugging: Check if prediction_data contains data
if prediction_data.empty:
    print("Error: prediction_data is empty. Check the time range or input data.")
else:
    #print(f"prediction_data contains {len(prediction_data)} rows.")
    pass


# Prepare the dataset for predictions
def prepare_sequences(data, input_columns, sequence_length):
    sequences = []
    for i in range(len(data) - sequence_length):
        seq = data[input_columns].iloc[i:i + sequence_length].values
        sequences.append(seq)
    return np.array(sequences)
x_data = prepare_sequences(prediction_data, input_columns, sequence_length)

# Debugging: Check if x_data contains sequences
if x_data.size == 0:
        print("Error: No sequences were generated. Check the sequence_length or input columns.")
else:
    print(f"Generated {len(x_data)} sequences for prediction.")

# Generate predictions
print("Generating predictions...")
predictions = []
with torch.no_grad():
    for seq in x_data:
        seq_tensor = torch.tensor(seq, dtype=torch.float32).unsqueeze(0)
        # to cuda
        seq_tensor = seq_tensor.to("cuda")
        pred = model(seq_tensor)
        # to cpu
        pred = pred.cpu()
        predictions.append(pred.squeeze(0).numpy())

# Debugging: Check if predictions were generated
if len(predictions) > 0:
    predictions = np.concatenate(predictions, axis=0)
    actuals = prediction_data[target_columns].values[:len(predictions)]

    # Prepare for plotting
    hour_range = prediction_data["hour"].values[:len(predictions)]
    pred_df = pd.DataFrame(predictions, columns=[f"pred_{col}" for col in target_columns])
    actual_df = pd.DataFrame(actuals, columns=[f"actual_{col}" for col in target_columns])
    hour_df = pd.DataFrame(hour_range, columns=["hour"])
    result_df = pd.concat([pred_df, actual_df, hour_df], axis=1)

    # Plot predictions vs actuals
    for column in target_columns:
        fig = px.line(
            result_df,
            x="hour",
            y=[f"pred_{column}", f"actual_{column}"],
            title=f"{column} - Predictions vs Actuals",
            labels={"hour": "Time", "value": "Value"},
            template="plotly_dark"
        )
        # Update x-axis tick format to include both date and time
        fig.update_xaxes(tickformat="%Y-%m-%d %H:%M")
        file_name = f"output/predicted_24h_{column}_interactive_plot.html"
        fig.write_html(file_name)
        print(f"Interactive plot saved: {file_name}")

else:
    print("Error: No predictions were generated. Check the model or input data.")