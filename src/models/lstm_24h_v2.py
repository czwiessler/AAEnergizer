import torch
import torch.nn as nn
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

# --- 1. Dataset-Klasse ---
class PowerDataset(Dataset):
    def __init__(self, data, input_columns, target_columns, sequence_length, forecast_horizon):
        self.features = data[input_columns].values
        self.targets = data[target_columns].values
        self.sequence_length = sequence_length
        self.forecast_horizon = forecast_horizon

    def __len__(self):
        # z.B. len(data) - sequence_length - forecast_horizon, damit wir genug Daten für das Target haben
        return len(self.features) - self.sequence_length - self.forecast_horizon + 1

    def __getitem__(self, idx):
        x = self.features[idx : idx + self.sequence_length]
        y = self.targets[idx + self.sequence_length : idx + self.sequence_length + self.forecast_horizon]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

# --- 2. Lightning-Modul ---
class UtilizationPredictor(pl.LightningModule):
    def __init__(self, input_size, hidden_size, num_layers, output_size, forecast_horizon, learning_rate):
        super(UtilizationPredictor, self).__init__()
        self.save_hyperparameters()  # optional, für Logging

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        # hier: output_size * forecast_horizon = Zahl der Target-Features * 24
        self.fc = nn.Linear(hidden_size, output_size * forecast_horizon)
        self.criterion = nn.MSELoss()
        self.learning_rate = learning_rate
        self.forecast_horizon = forecast_horizon

    def forward(self, x):
        # LSTM-Ausgabe
        out, _ = self.lstm(x)    # out: (batch_size, seq_length, hidden_size)
        # Nur das letzte Hidden State nehmen
        out = out[:, -1, :]      # (batch_size, hidden_size)
        # In "forecast_horizon * output_size" umwandeln
        out = self.fc(out)       # (batch_size, forecast_horizon * output_size)
        # relu to avoid negative values
        out = torch.relu(out)
        # Form anpassen: (batch_size, forecast_horizon, output_size)
        return out.view(-1, self.forecast_horizon, int(out.shape[1]/self.forecast_horizon))

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

# --- 3. Daten laden ---
df = pd.read_csv("data/processed/hourly_avg_power_cut.csv")
# One-Hot Kodierungen, falls du willst
df = pd.get_dummies(df, columns=['day_of_week', 'hour_of_day'])
# (a) Aufteilen in Train/Val/Test
train_size = int(len(df)*0.7)
val_size = int(len(df)*0.2)
test_size = len(df) - train_size - val_size

train_df = df.iloc[:train_size].copy()
val_df   = df.iloc[train_size:train_size+val_size].copy()
test_df  = df.iloc[train_size+val_size:].copy()

# save the test_df for later use
test_df.to_csv("data/processed/test_dataset_shifted.csv", index=False)

# (b) Spalten definieren
target_columns = [
    'avgChargingPower_site_1', 'activeSessions_site_1',
    'avgChargingPower_site_2', 'activeSessions_site_2'
]
# Beispiel-Feature-Spalten
# Du kannst/ solltest hier mehr Variablen einbauen (z. B. day_of_week, holiday, etc.)
#feature_columns = ['temperature', 'precipitation', 'hour_sin', 'hour_cos', 'is_holiday', 'is_weekend', 'is_vacation']
feature_columns = ['hour_sin', 'hour_cos', 'is_holiday', 'is_weekend', 'is_vacation']
#feature_columns = ['is_holiday', 'is_weekend', 'is_vacation']

# z.B.:
dummies = [col for col in df.columns if col.startswith('day_of_week_')]# or col.startswith('hour_of_day_')]
feature_columns += dummies

# (c) Skalierung (MinMaxScaler nur auf Trainingsdaten fitten!)
scaler_features = MinMaxScaler()
scaler_targets = MinMaxScaler()

train_df[feature_columns] = scaler_features.fit_transform(train_df[feature_columns])
val_df[feature_columns]   = scaler_features.transform(val_df[feature_columns])
test_df[feature_columns]  = scaler_features.transform(test_df[feature_columns])

train_df[target_columns] = scaler_targets.fit_transform(train_df[target_columns])
val_df[target_columns]   = scaler_targets.transform(val_df[target_columns])
test_df[target_columns]  = scaler_targets.transform(test_df[target_columns])

# print the fitted scaler so that we can use it later
import joblib
joblib.dump(scaler_features, 'models/scaler_features.pkl')
joblib.dump(scaler_targets, 'models/scaler_targets.pkl')

# (d) DataSets anlegen
sequence_length  = 24
forecast_horizon = 24  # ggf. 1, je nach gewünschter Vorhersage
train_dataset = PowerDataset(train_df, feature_columns, target_columns, sequence_length, forecast_horizon)
val_dataset   = PowerDataset(val_df,  feature_columns, target_columns, sequence_length, forecast_horizon)
test_dataset  = PowerDataset(test_df, feature_columns, target_columns, sequence_length, forecast_horizon)

# (e) DataLoader
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
val_loader   = DataLoader(val_dataset,   batch_size=32, shuffle=False)
test_loader  = DataLoader(test_dataset,  batch_size=32, shuffle=False)

# --- 4. Training starten ---
if __name__ == '__main__':
    import mlflow

    model = UtilizationPredictor(
        input_size=len(feature_columns),
        hidden_size=128,
        num_layers=2,
        output_size=len(target_columns),
        forecast_horizon=forecast_horizon,
        learning_rate=0.0001
    )
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath="models",
        filename="24h-model-{epoch:02d}-{val_loss:.5f}",
        save_top_k=1,
        mode="min",
    )
    trainer = pl.Trainer(
        max_epochs=100,         # deutlich mehr als 10
        log_every_n_steps=1,
        callbacks = [checkpoint_callback]
    )

    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, test_loader)
