import csv
import time
from rtde_receive import RTDEReceiveInterface

# DECLARE CONSTANTS
ROBOT_IP = "127.0.0.1"
SAMPLE_RATE_HZ = 100
DURATION_SEC = 5
OUTPUT_CSV = "../data/logs/joint_log.csv"

# CUSTOM RECIPE
def connect_with_recipe(ip: str, variables: list[str]) -> RTDEReceiveInterface:
    """Connect to the controller's receive interface with a custom output recipe."""
    try:
        rtde_r = RTDEReceiveInterface(ip, variables=variables)
    except RuntimeError as e:
        print(f"[connect] Failed to reach {ip}: {e}")
        raise   # this just throws exception and doesn't run whatever is after this line in connect_with_recipe function
    return rtde_r

RECIPE = ["timestamp", "actual_q"]


# CSV FILE FOR EASY DEBUGGING
def setup_csv(path: str, header: list[str]):
    """Open a CSV file for writing and write the header row"""
    f = open(path, mode="w", newline="")
    writer = csv.writer(f)
    writer.writerow(header)
    return f, writer

HEADER = ["timestamp", "q0", "q1", "q2", "q3", "q4", "q5"]


# Timed sampling loop for reading from controller at exactly 100Hz for 5 seconds and writing into CSV
def sample_loop(rtde_r: RTDEReceiveInterface, writer, rate_hz: int, duration_sec: int):
    """Sample joint state at rate+hz for duration_sec and write rows to CSV"""
    period = 1.0 / rate_hz
    n_samples = rate_hz * duration_sec

    for i in range(n_samples):
        start = time.time()

        timestamp = rtde_r.getTimestamp()
        q = rtde_r.getActualQ()
        writer.writerow([timestamp, *q])

        elapsed = time.time() - start
        sleep_time = period - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)


def cleanup(rtde_r: RTDEReceiveInterface, f):
    """Close the CSV file and disconnect from the controller."""
    f.close()
    rtde_r.disconnect()


if __name__ == "__main__":
    rtde_r = connect_with_recipe(ROBOT_IP, RECIPE)
    f, writer = setup_csv(OUTPUT_CSV, HEADER)
    try:
        print(f"Logging {SAMPLE_RATE_HZ} Hz for {DURATION_SEC}s to {OUTPUT_CSV}...")
        sample_loop(rtde_r, writer, SAMPLE_RATE_HZ, DURATION_SEC)
        print("Done.")
    finally:    # guaranteed to run below code even if try failed
        cleanup(rtde_r, f)

