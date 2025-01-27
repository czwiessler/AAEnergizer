import pandas as pd

def remove_single_rows_with_implausible_timestamps(dataset_path):
    df = pd.read_csv(dataset_path)

    #remove rows with implausible timestamps of over 1 minute difference
    ids_to_be_removed = [
        '5c99728ff9af8b5022123831',    # disconnecttime < donechargingtime
        '5e7954b0f9af8b090600ec84',    # disconnecttime < donechargingtime
        '5c2e85daf9af8b13dab07564',    # donechargingtime < connectiontime
        '5c2e85daf9af8b13dab07565',    # donechargingtime < connectiontime
        '5c2e85daf9af8b13dab07566'     # donechargingtime < connectiontime
    ]

    #remove all rows from ids_to_be_removed from df
    df = df[~df['sessionID'].isin(ids_to_be_removed)]

    df.to_csv(dataset_path, index=False)

if __name__ == "__main__":
    remove_single_rows_with_implausible_timestamps(dataset_path="data/processed/charging_sessions_cleaned.csv")
