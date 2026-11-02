import pandas as pd

LOG_PATH = "data/log.csv"

def load_log():
    """로그 파일 불러오기"""
    try:
        df = pd.read_csv(LOG_PATH, names=["text", "label", "prob", "timestamp"])
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df
    except FileNotFoundError:
        return pd.DataFrame(columns=["text", "label", "prob", "timestamp"])