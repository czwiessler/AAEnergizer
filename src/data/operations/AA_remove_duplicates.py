import pandas as pd

def remove_duplicates(dataset_path):
    df = pd.read_csv(dataset_path, parse_dates=['connectionTime', 'doneChargingTime', 'disconnectTime'])

    #remove duplicates based on 'id'
    df = df.drop_duplicates(subset=['id'], keep='first')

    df.to_csv(dataset_path, index=False)

    print("AA_: duplicates removed")


if __name__ == "__main__":
    remove_duplicates(dataset_path="data/processed/charging_sessions_cleaned.csv")