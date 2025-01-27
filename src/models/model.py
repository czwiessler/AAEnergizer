import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pandas as pd
import numpy as np

# -------------------------------------------------------
# 1) Daten laden
# -------------------------------------------------------
train_df = pd.read_csv("data/processed/split/train_set.csv")
val_df = pd.read_csv("data/processed/split/val_set.csv")
test_df = pd.read_csv("data/processed/split/test_set.csv")

# -------------------------------------------------------
# 2) Spalten definieren und Zielvariable
# -------------------------------------------------------
drop_columns = [
    "utilization",
    "utilization_lag_1h",
    "utilization_lag_24h",
    "active_sessions",
    "Unnamed: 0",
    "id",
    "sessionID"
]
target = "utilization"
categorical_features = ["siteID", "stationID", "timezone"]


# -------------------------------------------------------
# 3) Hilfsfunktionen
# -------------------------------------------------------
def add_datetime_features(df: pd.DataFrame, source_col: str = "connectionTime") -> pd.DataFrame:
    """
    Konvertiert source_col zu datetime und extrahiert hour, day, month, weekday.
    """
    if source_col in df.columns:
        df[source_col] = pd.to_datetime(df[source_col], errors="coerce")
        df["hour"] = df[source_col].dt.hour
        df["day"] = df[source_col].dt.day
        df["month"] = df[source_col].dt.month
        df["weekday"] = df[source_col].dt.weekday
    return df


def encode_cyclical_feature(df: pd.DataFrame, col: str, max_val: int) -> pd.DataFrame:
    """ Wandelt Spalte col in sin/cos um und droppt col. """
    if col in df.columns:
        df[f"{col}_sin"] = np.sin(2 * np.pi * df[col] / max_val)
        df[f"{col}_cos"] = np.cos(2 * np.pi * df[col] / max_val)
        df.drop(columns=[col], inplace=True)
    return df


def remove_zero_std_columns(df: pd.DataFrame, features: list) -> list:
    """
    Entfernt alle Feature-Spalten mit Standardabweichung == 0 in df
    und gibt die bereinigte Feature-Liste zurück.
    """
    desc = df[features].describe()
    zero_std_cols = desc.columns[desc.loc["std"] == 0.0].tolist()
    if zero_std_cols:
        print(">>> Achtung: Folgende Spalten haben 0-Varianz und werden entfernt:", zero_std_cols)
        for col in zero_std_cols:
            if col in features:
                features.remove(col)
    return features


# -------------------------------------------------------
# 4) Vorverarbeitung TRAIN
# -------------------------------------------------------
# 4.1) DateTime-Features
train_df = add_datetime_features(train_df, source_col="connectionTime")

# 4.2) Kategorische Spalten (nur auf TRAIN fitten)
encoders = {}
for col in categorical_features:
    if col in train_df.columns:
        le = LabelEncoder()
        train_df[col] = le.fit_transform(train_df[col].astype(str))
        encoders[col] = le

# 4.3) Zyklische Features: hour, day, month, weekday
for cyc_col, max_val in [("hour", 24), ("day", 31), ("month", 12), ("weekday", 7)]:
    train_df = encode_cyclical_feature(train_df, cyc_col, max_val)

# 4.4) Feature-Liste = alle numerischen Spalten außer target + Drop-Spalten
train_numeric_cols = [
    c for c in train_df.columns
    if c not in drop_columns + [target]
       and pd.api.types.is_numeric_dtype(train_df[c])
]
features = train_numeric_cols

# 4.5) Check auf 0-Varianz-Spalten in TRAIN
features = remove_zero_std_columns(train_df, features)

# 4.6) Check auf NaNs / Infs in TRAIN
print("\n>>> Checking for NaNs in TRAIN...")
print(train_df[features + [target]].isna().sum())

print("\n>>> Checking for Infs in TRAIN...")
inf_mask = np.isinf(train_df[features + [target]]).sum()
print(inf_mask)

# 4.7) Inf durch NaN ersetzen, dann na droppen
train_df.replace([np.inf, -np.inf], np.nan, inplace=True)
train_df.dropna(subset=features + [target], inplace=True)

# 4.8) StandardScaler: nur auf TRAIN fit
scaler = StandardScaler()
train_df[features] = scaler.fit_transform(train_df[features])


# -------------------------------------------------------
# 5) Vorverarbeitung VAL/TEST (gleiche Schritte, aber transform)
# -------------------------------------------------------
def apply_preprocessing_for_val_and_test(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # (a) DateTime-Features
    df = add_datetime_features(df, source_col="connectionTime")
    # (b) Kategorische Enkodierung
    for col, le in encoders.items():
        if col in df.columns:
            df[col] = df[col].astype(str)
            df[col] = le.transform(df[col])
    # (c) Zyklische Features
    for cyc_col, max_val in [("hour", 24), ("day", 31), ("month", 12), ("weekday", 7)]:
        df = encode_cyclical_feature(df, cyc_col, max_val)
    # (d) Inf -> NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    # (e) Jetzt NaNs droppen (ohne errors-Argument)
    df.dropna(subset=features + [target], inplace=True)
    # (f) Skalierung (nur Spalten, die in features vorhanden sind)
    existing = [f for f in features if f in df.columns]
    if existing:
        df[existing] = scaler.transform(df[existing])
    return df


val_df = apply_preprocessing_for_val_and_test(val_df)
test_df = apply_preprocessing_for_val_and_test(test_df)

# -------------------------------------------------------
# 6) Debug-Prints nach Preprocessing
# -------------------------------------------------------
print("\n>>> TRAIN shape:", train_df.shape)
print(">>> VAL shape:", val_df.shape)
print(">>> TEST shape:", test_df.shape)


# -------------------------------------------------------
# 7) PyTorch Dataset + DataLoader
# -------------------------------------------------------
class UtilizationDataset(Dataset):
    def __init__(self, data: pd.DataFrame, target: str):
        self.features = [f for f in features if f in data.columns]
        self.X = data[self.features].values.astype(np.float32)
        self.y = data[target].values.astype(np.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


train_dataset = UtilizationDataset(train_df, target)
val_dataset = UtilizationDataset(val_df, target)
test_dataset = UtilizationDataset(test_df, target)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


# -------------------------------------------------------
# 8) Neuronales Netz
# -------------------------------------------------------
class UtilizationModel(nn.Module):
    def __init__(self, input_dim):
        super(UtilizationModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.model(x).squeeze()


# -------------------------------------------------------
# 9) Trainingseinstellungen
# -------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = UtilizationModel(input_dim=len(train_dataset.features)).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# -------------------------------------------------------
# 10) Trainingsfunktion
# -------------------------------------------------------
def train_model(model, train_loader, val_loader, criterion, optimizer, epochs=20):
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * X_batch.size(0)

        # Validierung
        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item() * X_batch.size(0)

        train_loss /= len(train_loader.dataset)
        val_loss /= len(val_loader.dataset)

        print(f"Epoch {epoch + 1:02d}/{epochs:02d} - "
              f"Train Loss: {train_loss:.4f}, "
              f"Val Loss: {val_loss:.4f}")


# -------------------------------------------------------
# 11) Modell trainieren
# -------------------------------------------------------
train_model(model, train_loader, val_loader, criterion, optimizer, epochs=20)

# -------------------------------------------------------
# 11a) Modell speichern
# Speichere das Modell unter models/model.pth ab
torch.save(model.state_dict(), "models/model.pth")
# -------------------------------------------------------

# -------------------------------------------------------
# 12) Modell testen
# -------------------------------------------------------
model.eval()
predictions = []
with torch.no_grad():
    for X_batch, _ in test_loader:
        X_batch = X_batch.to(device)
        outputs = model(X_batch)
        predictions.extend(outputs.cpu().numpy())

# -------------------------------------------------------
# 13) Ergebnisse in Test-DataFrame
# -------------------------------------------------------
test_df["predicted_utilization"] = predictions

print("\nVorhersagen (erste 10 Zeilen):")
print(test_df[["predicted_utilization"]].head(10))

# -------------------------------------------------------
# 14) Optional: Vorhersagen als CSV speichern
# -------------------------------------------------------
test_df.to_csv("output/predictions.csv", index=False)
print("\nVorhersagen wurden als 'predictions.csv' gespeichert.")
