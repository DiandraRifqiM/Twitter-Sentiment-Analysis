# Lib Import
import datetime as dt

def log_report(model_name, report, log_path=r"logs\train.log"):
    with open(log_path, "a") as f:
        time = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"[{time}] {model_name} Classification Report:\n")
        f.write(report + "\n")
        f.write("="*50 + "\n")


