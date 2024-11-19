import pandas as pd

pd.read_csv("../../data/charging_sessions.csv").to_csv("../../data/charging_sessions_cleaned.csv", index=False)