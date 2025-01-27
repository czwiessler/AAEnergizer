import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
from pytorch_lightning.utilities.rank_zero import rank_zero_only

# --- Dataset Class ---
class PowerDataset(Dataset):
    def __init__(self, data, input_columns, target_columns, sequence_length, forecast_horizon):
        self.features = data[input_columns].values
        self.targets = data[target_columns].values
        self.sequence_length = sequence_length
        self.forecast_horizon = forecast_horizon

    def __len__(self):
        return len(self.features) - self.sequence_length - self.forecast_horizon

    def __getitem__(self, idx):
        x = self.features[idx:idx + self.sequence_length]
        y = self.targets[idx + self.sequence_length: idx + self.sequence_length + self.forecast_horizon]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

# --- Lightning Module ---
class UtilizationPredictor(pl.LightningModule):
    def __init__(self, input_size, hidden_size, num_layers, output_size, forecast_horizon, learning_rate):
        super(UtilizationPredictor, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size * forecast_horizon)
        self.criterion = nn.MSELoss()
        self.learning_rate = learning_rate
        self.forecast_horizon = forecast_horizon

    def forward(self, x):
        out, _ = self.lstm(x)  # out hat die Form (batch_size, seq_length, hidden_size)
        out = self.fc(
            out[:, -1, :])  # nur das letzte Hidden-State verwenden; Form: (batch_size, output_size * forecast_horizon)
        batch_size = out.size(0)
        return out.view(batch_size, self.forecast_horizon, -1)  # explizite Batchgröße

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('val_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('test_loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

# --- Data Preparation ---
data_path = 'data/processed/hourly_avg_power_cut.csv'
df = pd.read_csv(data_path)

# Define input and target columns
input_columns_old = [
    'temperature', 'precipitation', 'is_holiday', 'is_weekend', 'is_vacation',
    'hour_sin', 'hour_cos', 'season_sin', 'season_cos'
]
input_columns = [
    'day_of_week', 'hour_of_day'
]

target_columns = [
    'avgChargingPower_site_1', 'activeSessions_site_1',
    'avgChargingPower_site_2', 'activeSessions_site_2'
]

# Normalize the data
scaler_features = MinMaxScaler()
scaler_targets = MinMaxScaler()

# one hot encoding for hour_of_day and day_of_week
df = pd.get_dummies(df, columns=['hour_of_day', 'day_of_week'])
input_columns = [col for col in df.columns if 'hour_of_day_' in col or 'day_of_week_' in col]

#df[input_columns] = scaler_features.fit_transform(df[input_columns])
#df[target_columns] = scaler_targets.fit_transform(df[target_columns])

# Parameters for Dataset
sequence_length = 24  # Input length: 24 hours
forecast_horizon = 1  # Forecast for the next 24 hours
batch_size = 128

train_size = int(len(df) * 0.7)
val_size = int(len(df) * 0.2)
test_size = len(df) - train_size - val_size

train_data = df[:train_size]
val_data = df[train_size:train_size + val_size]
test_data = df[train_size + val_size:]

train_dataset = PowerDataset(train_data, input_columns, target_columns, sequence_length, forecast_horizon)
val_dataset = PowerDataset(val_data, input_columns, target_columns, sequence_length, forecast_horizon)
test_dataset = PowerDataset(test_data, input_columns, target_columns, sequence_length, forecast_horizon)
# save the test_dataset "data/processed/test_dataset.csv"
test_data.to_csv("data/processed/test_dataset.csv", index=False)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=1)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=1)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=1)

# --- Training with PyTorch Lightning ---
if __name__ == '__main__':
    import mlflow
    import mlflow.pytorch

    input_size = len(input_columns)
    hidden_size = 128
    num_layers = 2
    output_size = len(target_columns)
    learning_rate = 0.001

    mlflow.set_experiment("Power Utilization Prediction - 24 Hours")

    with mlflow.start_run() as run:
        model = UtilizationPredictor(input_size, hidden_size, num_layers, output_size, forecast_horizon, learning_rate)

        checkpoint_callback = ModelCheckpoint(
            monitor="val_loss",
            dirpath="models",
            filename="24h-model-{epoch:02d}-{val_loss:.5f}",
            save_top_k=1,
            mode="min",
        )

        trainer = pl.Trainer(
            max_epochs=10,
            log_every_n_steps=1,
            callbacks=[checkpoint_callback],
        )

        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

        test_metrics = trainer.test(model, dataloaders=test_loader, verbose=False)
        for metric in test_metrics:
            for key, value in metric.items():
                mlflow.log_metric(key, value)

        mlflow.pytorch.log_model(model, "model")
        print(f"Model saved in run: {run.info.run_id}")
