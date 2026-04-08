from collections import Counter
from datetime import datetime

import pandas as pd

LOG_FILE = "system_logs.txt"


def parse_log_line(line: str):
    """
    Convert one raw log line -> (timestamp, level, message)
    Example line format:
    2025-03-27 10:00:00 INFO Suspicious IP access blocked
    """
    line = line.strip()
    if not line:
        return None

    parts = line.split()

    # Expect at least: date, time, level, message...
    if len(parts) < 4:
        return None

    date_str = parts[0]              # 2025-03-27
    time_str = parts[1]              # 10:00:00
    level = parts[2]                 # INFO / WARNING / ERROR / CRITICAL
    message = " ".join(parts[3:])    # rest of the line

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


def find_error_bursts(df: pd.DataFrame, window_seconds: int = 30, threshold: int = 4):
    """
    Find time windows where ERROR count is >= threshold inside given seconds.
    """
    error_times = df[df["level"] == "ERROR"]["timestamp"]

    # floor timestamps to window (e.g. 30 seconds)
    error_counts = Counter(error_times.dt.floor(f"{window_seconds}s"))

    anomalies = []
    for ts, count in error_counts.items():
        if count >= threshold:
            anomalies.append((ts, count))

    return sorted(anomalies, key=lambda x: x[0])


def main():
    df = load_logs(LOG_FILE)

    print("\nFull log preview:")
    print(df.head())
    print(f"\nTotal log lines: {len(df)}")

    anomalies = find_error_bursts(df, window_seconds=30, threshold=4)

    print("\n🔍 Detected ERROR bursts (>=4 errors in 30 seconds):")
    if not anomalies:
        print("No bursts found.")
    else:
        for ts, count in anomalies:
            print(f"🚨 {count} ERROR logs around {ts}")


if __name__ == "__main__":
    main()