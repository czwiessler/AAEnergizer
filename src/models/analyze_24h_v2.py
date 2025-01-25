import torch
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from torch.utils.data import DataLoader
from src.models.lstm_24h_v2 import UtilizationPredictor, PowerDataset


def prepare_24h_batches(df, start_hour=4):
    """
    Teilt das Dataset in 24-Stunden-Abschnitte auf, beginnend bei `start_hour`.
    """
    df['hour_of_day'] = df['hour'].dt.hour
    start_indices = df[df['hour_of_day'] == start_hour].index

    batches = []
    for start_idx in start_indices:
        if start_idx + 24 <= len(df):
            batch = df.iloc[start_idx:start_idx + 24]
            batches.append(batch)

    return batches


def generate_plots_for_24h_predictions(
        ckpt_path: str,
        test_batches,
        target_columns,
        input_columns,
        device: str = "cpu"
):
    """
    Macht Vorhersagen für 24h-Batches und erstellt ein gemeinsames Diagramm.
    """
    # Modell vom Checkpoint laden
    model = UtilizationPredictor.load_from_checkpoint(ckpt_path)
    model.to(device)
    model.eval()

    # Initialisierung für alle Predictions und Stunden
    all_predictions = []
    all_actuals = []
    all_hours = []

    # Vorhersagen pro Batch
    with torch.no_grad():
        for batch in test_batches:
            # Daten vorbereiten
            input_data = batch[input_columns].values
            targets = batch[target_columns].values
            hours = batch["hour"].values

            # Batch in Tensor konvertieren
            input_tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0).to(device)
            target_tensor = torch.tensor(targets, dtype=torch.float32).unsqueeze(0).to(device)

            # Modellvorhersagen
            predictions = model(input_tensor).cpu().numpy()

            # Ergebnisse sammeln
            all_predictions.append(predictions[0])  # Batch (24, num_targets)
            all_actuals.append(targets)  # (24, num_targets)
            all_hours.append(hours)  # (24,)

    # DataFrames erstellen
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_actuals = np.concatenate(all_actuals, axis=0)
    all_hours = np.concatenate(all_hours, axis=0)

    col_pred = [f"pred_{col}" for col in target_columns]
    col_act = [f"actual_{col}" for col in target_columns]

    df_predictions = pd.DataFrame(all_predictions, columns=col_pred)
    df_actuals = pd.DataFrame(all_actuals, columns=col_act)
    df_combined = pd.concat([df_predictions, df_actuals], axis=1)
    df_combined['hour'] = all_hours

    # Plotly-Diagramm erstellen
    fig = go.Figure()

    for col in target_columns:
        fig.add_trace(go.Scatter(
            x=df_combined["hour"],
            y=df_combined[f"pred_{col}"],
            mode="lines",
            name=f"Prediction: {col}"
        ))
        fig.add_trace(go.Scatter(
            x=df_combined["hour"],
            y=df_combined[f"actual_{col}"],
            mode="lines",
            name=f"Actual: {col}"
        ))

    fig.update_layout(
        title="24h Predictions vs. Actuals",
        xaxis_title="Hour",
        yaxis_title="Value",
        legend_title="Legend"
    )
    fig.show()
    # save the plot
    fig.write_html("output/24h_predictions.html")


if __name__ == "__main__":
    # --- Parameter ---
    target_columns = [
        'avgChargingPower_site_1',
        'activeSessions_site_1',
        'avgChargingPower_site_2',
        'activeSessions_site_2'
    ]
    input_columns = ['temperature', 'precipitation', 'hour_sin', 'hour_cos']
    dummies = ['day_of_week_0', 'day_of_week_1', 'day_of_week_2', 'day_of_week_3', 'day_of_week_4', 'day_of_week_5', 'day_of_week_6', 'hour_of_day_0', 'hour_of_day_1', 'hour_of_day_2', 'hour_of_day_3', 'hour_of_day_4', 'hour_of_day_5', 'hour_of_day_6', 'hour_of_day_7', 'hour_of_day_8', 'hour_of_day_9', 'hour_of_day_10', 'hour_of_day_11', 'hour_of_day_12', 'hour_of_day_13', 'hour_of_day_14', 'hour_of_day_15', 'hour_of_day_16', 'hour_of_day_17', 'hour_of_day_18', 'hour_of_day_19', 'hour_of_day_20', 'hour_of_day_21', 'hour_of_day_22', 'hour_of_day_23']
    input_columns += dummies
    sequence_length = 24

    # Testdaten laden
    df_test = pd.read_csv("data/processed/test_dataset.csv", parse_dates=["hour"])
    # Skalieren (Laden der Scaler)
    import joblib

    scaler_features = joblib.load('models/scaler_features.pkl')
    scaler_targets = joblib.load('models/scaler_targets.pkl')
    df_test[input_columns] = scaler_features.transform(df_test[input_columns])
    df_test[target_columns] = scaler_targets.transform(df_test[target_columns])

    # 24h-Batches vorbereiten
    test_batches = prepare_24h_batches(df_test, start_hour=4)

    # Checkpoint-Pfad
    ckpt_path = "models/24h-model-epoch=09-val_loss=0.01461.ckpt"

    # Vorhersagen und Plot generieren
    generate_plots_for_24h_predictions(
        ckpt_path=ckpt_path,
        test_batches=test_batches,
        target_columns=target_columns,
        input_columns=input_columns,
        device="cpu"
    )
