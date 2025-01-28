import pandas as pd

def copy_csv(from_path, to_path):
    pd.read_csv(from_path).to_csv(to_path, index=False)



if __name__ == "__main__":
    copy_csv(from_path="data/raw/charging_sessions.csv", to_path="data/processed/charging_sessions_cleaned.csv")