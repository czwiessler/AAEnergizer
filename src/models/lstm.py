import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.tuner import Tuner
from sklearn.preprocessing import MinMaxScaler
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.rank_zero import rank_zero_only

# --- Dataset Class ---
class PowerDataset(Dataset):
    def __init__(self, data, input_columns, target_columns, sequence_length):
        self.features = data[input_columns].values
        self.targets = data[target_columns].values
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.features) - self.sequence_length

    def __getitem__(self, idx):
        x = self.features[idx:idx + self.sequence_length]
        y = self.targets[idx + self.sequence_length]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

# --- Lightning Module ---
class UtilizationPredictor(pl.LightningModule):
    def __init__(self, input_size, hidden_size, num_layers, output_size, learning_rate):
        super(UtilizationPredictor, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.criterion = nn.MSELoss()
        self.learning_rate = learning_rate
        self.test_predictions = []
        self.test_actuals = []
        self.train_losses = []  # Speicher für train_loss jeder Epoche
        self.val_losses = []    # Speicher für val_loss jeder Epoche

        # Placeholders for DataLoaders
        self._train_loader = None
        self._val_loader = None
        self._test_loader = None

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

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

        self.test_predictions.append(y_hat.detach().cpu().numpy())
        self.test_actuals.append(y.detach().cpu().numpy())

        self.log('test_loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def train_dataloader(self):
        return self._train_loader

    def val_dataloader(self):
        return self._val_loader

    def test_dataloader(self):
        return self._test_loader

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self._train_loader = train_loader
            self._val_loader = val_loader
        if stage == 'test' or stage is None:
            self._test_loader = test_loader

    def on_test_end(self):
        # Konvertiere Listen in numpy-Arrays
        predictions = np.concatenate(self.test_predictions, axis=0)
        actuals = np.concatenate(self.test_actuals, axis=0)

        # Speichere die Arrays in .npy-Dateien
        np.save('src/models/test_predictions.npy', predictions)
        np.save('src/models/test_actuals.npy', actuals)

        print("Predictions and actuals saved as 'test_predictions.npy' and 'test_actuals.npy'")

    @rank_zero_only
    def on_train_epoch_end(self):
        # Train Loss
        train_loss = self.trainer.callback_metrics.get("train_loss")
        if train_loss is not None:
            train_loss = train_loss.item()
            self.train_losses.append(train_loss)
            mlflow.log_metric("epoch_train_loss", train_loss, step=self.current_epoch)

        # Validation Loss
        val_loss = self.trainer.callback_metrics.get("val_loss")
        if val_loss is not None:
            val_loss = val_loss.item()
            self.val_losses.append(val_loss)
            mlflow.log_metric("epoch_val_loss", val_loss, step=self.current_epoch)
            print(f"\nEpoch {self.current_epoch}: Train Loss: {train_loss}, Validation Loss: {val_loss}")

# --- Data Preparation ---
# Load the data
data_path = 'data/processed/hourly_avg_power_cut.csv'
df = pd.read_csv(data_path)

# Define input and target columns
input_columns = [
    'temperature', 'precipitation', 'is_holiday', 'is_weekend', 'is_vacation',
    'hour_sin', 'hour_cos', 'season_sin', 'season_cos'
]
target_columns = [
    'avgChargingPower_site_1', 'activeSessions_site_1',
    'avgChargingPower_site_2', 'activeSessions_site_2'
]

# Normalize the data
scaler_features = MinMaxScaler()
scaler_targets = MinMaxScaler()

df[input_columns] = scaler_features.fit_transform(df[input_columns])
df[target_columns] = scaler_targets.fit_transform(df[target_columns])

# Create Dataset and DataLoader
sequence_length = 24  # Use 24 hours of data for prediction
batch_size = 32

train_size = int(len(df) * 0.7)
val_size = int(len(df) * 0.2)
test_size = len(df) - train_size - val_size

train_data = df[:train_size]
val_data = df[train_size:train_size + val_size]
test_data = df[train_size + val_size:]

train_dataset = PowerDataset(train_data, input_columns, target_columns, sequence_length)
val_dataset = PowerDataset(val_data, input_columns, target_columns, sequence_length)
test_dataset = PowerDataset(test_data, input_columns, target_columns, sequence_length)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=1)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=1)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=1)

# --- Training with PyTorch Lightning ---
if __name__ == '__main__':
    import mlflow
    import mlflow.pytorch

    # --- Training mit PyTorch Lightning und MLflow-Integration ---
    # Model parameters
    input_size = len(input_columns)
    hidden_size = 64
    num_layers = 2
    output_size = len(target_columns)
    learning_rate = 0.0001

    # Starte ein MLflow-Experiment
    mlflow.set_experiment("Power Utilization Prediction")

    with mlflow.start_run() as run:
        # Logge die Modellparameter
        mlflow.log_param("input_size", input_size)
        mlflow.log_param("hidden_size", hidden_size)
        mlflow.log_param("num_layers", num_layers)
        mlflow.log_param("output_size", output_size)
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("sequence_length", sequence_length)
        mlflow.log_param("batch_size", batch_size)

        # Modell initialisieren
        model = UtilizationPredictor(input_size, hidden_size, num_layers, output_size, learning_rate)

        # Checkpoint-Callback für Modell-Speicherung
        checkpoint_callback = ModelCheckpoint(
            monitor="val_loss",
            dirpath="models",
            filename="model-{epoch:02d}-{val_loss:.5f}",
            save_top_k=1,
            mode="min",
        )

        # Trainer initialisieren
        trainer = pl.Trainer(
            max_epochs=5,
            log_every_n_steps=1,
            callbacks=[checkpoint_callback],
        )

        use_lr_finder = False
        if use_lr_finder:
            tuner = Tuner(trainer)
            lr_finder = tuner.lr_find(model)
            fig = lr_finder.plot(suggest=True)
            fig.show()

        # Training
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

        # Validation-Ergebnisse loggen
        metrics = trainer.validate(model, dataloaders=val_loader, verbose=False)
        for metric in metrics:
            for key, value in metric.items():
                mlflow.log_metric(key, value)

        # Testen
        test_metrics = trainer.test(model, dataloaders=test_loader, verbose=False)
        for metric in test_metrics:
            for key, value in metric.items():
                mlflow.log_metric(key, value)

        # Speichere das Modell in MLflow
        mlflow.pytorch.log_model(model, "model")
        print(f"Model saved in run: {run.info.run_id}")
