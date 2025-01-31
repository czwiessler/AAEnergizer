import pandas as pd

def cut_ds(nn_dataset_path, start_date, end_date):
    # Lade den Datensatz
    dataset = pd.read_csv(nn_dataset_path)

    # Stelle sicher, dass die 'hour'-Spalte als Datetime erkannt wird
    dataset['hour'] = pd.to_datetime(dataset['hour'])

    # Filtere den Datensatz basierend auf Start- und Enddatum
    filtered_dataset = dataset[(dataset['hour'] >= start_date) & (dataset['hour'] <= end_date)]

    # save the filtered dataset as a new csv file with same name but ending with _cut
    filtered_dataset.to_csv(nn_dataset_path.replace('.csv', '_cut.csv'), index=False)

    print("Dataset cut completed. Saved to ", nn_dataset_path.replace('.csv', '_cut.csv'))

    return filtered_dataset

if __name__ == "__main__":
    cut_ds("data/processed/hourly_avg_power.csv", "2018-09-05", "2020-03-01")