import pandas as pd
import numpy as np
from pmdarima import auto_arima
from sklearn.metrics import mean_squared_error, mean_absolute_error
import mlflow
import mlflow.sklearn
import joblib
from sklearn.preprocessing import MinMaxScaler

# Load data with datetime index
data_path = 'data/processed/hourly_avg_power_cut.csv'
df = pd.read_csv(data_path, parse_dates=['datetime'], index_col='datetime')

# Define input and target columns (same as LSTM)
input_columns = [
    'temperature', 'precipitation', 'is_holiday',
    'is_weekend', 'is_vacation', 'hour_sin', 'hour_cos',
    'season_sin', 'season_cos'
]
target_columns = [
    'avgChargingPower_site_1', 'activeSessions_site_1',
    'avgChargingPower_site_2', 'activeSessions_site_2'
]

# Normalization (same as LSTM)
scaler_features = MinMaxScaler()
scaler_targets = MinMaxScaler()

df[input_columns] = scaler_features.fit_transform(df[input_columns])
df[target_columns] = scaler_targets.fit_transform(df[target_columns])

# Split data into train, val, test (chronological split)
train_size = int(len(df) * 0.7)
val_size = int(len(df) * 0.2)
test_size = len(df) - train_size - val_size

train_data = df.iloc[:train_size]
val_data = df.iloc[train_size:train_size + val_size]
test_data = df.iloc[train_size + val_size:]

# MLflow setup
mlflow.set_experiment("Power Utilization Prediction - ARIMA")

for target_col in target_columns:
    with mlflow.start_run(run_name=f"ARIMA_{target_col}"):
        # Extract target series with datetime index
        train_series = train_data[target_col]
        val_series = val_data[target_col]
        test_series = test_data[target_col]

        # Auto ARIMA to find optimal parameters
        model = auto_arima(
            train_series,
            seasonal=False,
            trace=True,
            error_action='ignore',
            suppress_warnings=True,
            stepwise=True
        )

        # Log model parameters
        mlflow.log_params({
            "order": model.order,
            "seasonal_order": model.seasonal_order,
            "target": target_col
        })

        # Rolling forecast for validation
        val_predictions = []
        current_model = model

        # Predict validation set step-by-step
        for i in range(len(val_series)):
            pred = current_model.predict(n_periods=1)
            val_predictions.append(pred[0])
            current_model = current_model.update([val_series.iloc[i]])

        # Predict test set step-by-step
        test_predictions = []
        for i in range(len(test_series)):
            pred = current_model.predict(n_periods=1)
            test_predictions.append(pred[0])
            current_model = current_model.update([test_series.iloc[i]])

        # Convert to arrays
        val_pred = np.array(val_predictions)
        test_pred = np.array(test_predictions)

        # Inverse scaling
        target_idx = target_columns.index(target_col)
        min_val = scaler_targets.min_[target_idx]
        scale_val = scaler_targets.scale_[target_idx]

        val_pred_inv = (val_pred * scale_val) + min_val
        test_pred_inv = (test_pred * scale_val) + min_val

        val_actual_inv = (val_series.values * scale_val) + min_val
        test_actual_inv = (test_series.values * scale_val) + min_val

        # Calculate metrics
        val_mse = mean_squared_error(val_actual_inv, val_pred_inv)
        val_mae = mean_absolute_error(val_actual_inv, val_pred_inv)
        val_rmse = np.sqrt(val_mse)

        test_mse = mean_squared_error(test_actual_inv, test_pred_inv)
        test_mae = mean_absolute_error(test_actual_inv, test_pred_inv)
        test_rmse = np.sqrt(test_mse)

        # Log metrics
        mlflow.log_metrics({
            "val_mse": val_mse,
            "val_mae": val_mae,
            "val_rmse": val_rmse,
            "test_mse": test_mse,
            "test_mae": test_mae,
            "test_rmse": test_rmse
        })

        # Save model
        model_path = f"models/arima_{target_col.replace(' ', '_')}.pkl"
        joblib.dump(model, model_path)
        mlflow.log_artifact(model_path)

        # Print results
        print(f"\nResults for {target_col}:")
        print(f"Validation RMSE: {val_rmse:.4f}")
        print(f"Test RMSE: {test_rmse:.4f}")
        print(f"Best ARIMA order: {model.order}")

print("\nAll ARIMA models trained and logged in MLflow!")