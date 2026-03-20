import os
import time
import datetime

CHECKPOINT_PATH = "artifacts/models/best_model.pt"

def monitor():
    print(f"Monitoring {CHECKPOINT_PATH}...")
    if not os.path.exists(CHECKPOINT_PATH):
        print("Waiting for first checkpoint...")
    
    last_mtime = None
    if os.path.exists(CHECKPOINT_PATH):
        last_mtime = os.path.getmtime(CHECKPOINT_PATH)
        print(f"Initial checkpoint found. Last updated: {datetime.datetime.fromtimestamp(last_mtime)}")

    while True:
        try:
            if os.path.exists(CHECKPOINT_PATH):
                current_mtime = os.path.getmtime(CHECKPOINT_PATH)
                if last_mtime is None or current_mtime > last_mtime:
                    last_mtime = current_mtime
                    timestamp = datetime.datetime.fromtimestamp(current_mtime).strftime('%Y-%m-%d %H:%M:%S')
                    print(f"[{timestamp}] New best model saved! An epoch just finished and improved the Macro AUROC.")
            
            time.sleep(10)
        except KeyboardInterrupt:
            print("\nMonitoring stopped.")
            break

if __name__ == "__main__":
    monitor()
