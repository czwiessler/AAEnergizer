import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load predictions and actuals
predictions = np.load('test_predictions.npy')
actuals = np.load('test_actuals.npy')

# Define target columns
target_columns = [
   'avgChargingPower_site_1', 'activeSessions_site_1',
   'avgChargingPower_site_2', 'activeSessions_site_2'
]

# Prepare DataFrame for comparison
df_preds = pd.DataFrame(predictions, columns=[f"pred_{col}" for col in target_columns])
df_actuals = pd.DataFrame(actuals, columns=[f"actual_{col}" for col in target_columns])
df_compare = pd.concat([df_preds, df_actuals], axis=1)

# Print the first few rows of the combined DataFrame
print(df_compare.head())

# Create subplots
num_columns = len(target_columns)
fig, axes = plt.subplots(num_columns, 1, figsize=(12, 4 * num_columns), sharex=True)

# Plot each column pair in a separate subplot
for i, col in enumerate(target_columns):
    ax = axes[i] if num_columns > 1 else axes  # Handle single subplot case
    ax.plot(df_preds[f"pred_{col}"], label=f"Prediction: {col}")
    ax.plot(df_actuals[f"actual_{col}"], label=f"Actual: {col}")
    ax.set_title(f"{col} - Predictions vs Actuals")
    ax.legend()
    ax.grid()

# Adjust layout and display the plot
plt.tight_layout()
plt.show()
