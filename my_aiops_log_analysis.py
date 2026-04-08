from datetime import datetime

import pandas as pd
from sklearn.ensemble import IsolationForest

LOG_FILE = "system_logs.txt"


def parse_log_line(line: str):
    """
    Same format as before:
    2025-03-27 10:00:00 INFO Suspicious IP access blocked
    """
    line = line.strip()
    if not line:
        return None

    parts = line.split()
    if len(parts) < 4:
        return None

    date_str = parts[0]
    time_str = parts[1]
    level = parts[2]
    message = " ".join(parts[3:])

    ts_str = f"{date_str} {time_str}"

    try:
        timestamp = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return None

    return timestamp, level, message


def load_logs(filepath: str) -> pd.DataFrame:
    rows = []

    with open(filepath, "r") as f:
        for line in f:
            parsed = parse_log_line(line)
            if parsed is None:
                continue
            timestamp, level, message = parsed
            rows.append({"timestamp": timestamp, "level": level, "message": message})

    df = pd.DataFrame(rows)
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create numeric features for the model:
    - level_score
    - message_length
    """
    level_map = {"INFO": 1, "WARNING": 2, "ERROR": 3, "CRITICAL": 4}
    df["level_score"] = df["level"].map(level_map).fillna(0).astype(int)
    df["message_length"] = df["message"].str.len().astype(int)
    return df


def detect_anomalies(df: pd.DataFrame, contamination: float = 0.08) -> pd.DataFrame:
    """
    Use Isolation Forest to flag unusual log lines.
    contamination ~ fraction of data that is anomalous.
    """
    features = df[["level_score", "message_length"]]

    model = IsolationForest(
        n_estimators=100,
        contamination=contamination,
        random_state=42,
    )

    model.fit(features)
    preds = model.predict(features)  # 1 = normal, -1 = anomaly

    df["anomaly"] = preds
    df["is_anomaly"] = df["anomaly"].map({1: "✅ Normal", -1: "❌ Anomaly"})
    return df


def main():
    df = load_logs(LOG_FILE)
    df = build_features(df)
    df = detect_anomalies(df, contamination=0.08)

    print("\nSample of all logs with features:")
    print(df.head())

    anomalies = df[df["anomaly"] == -1]

    print("\n🔍 Detected Anomalies:")
    print(anomalies[["timestamp", "level", "message", "level_score", "message_length", "is_anomaly"]].head(20))
    print(f"\nTotal anomalies: {len(anomalies)} out of {len(df)} logs")


if __name__ == "__main__":
    main()