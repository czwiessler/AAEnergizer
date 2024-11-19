import pandas as pd

def copy_csv():
    pd.read_csv("../../data/charging_sessions.csv").to_csv("../../data/charging_sessions_cleaned.csv", index=False)




copy_csv()