import sys
import os
import numpy as np


def load_ms25_timestamps(ms25_path):
    """
    ms25.csv format: utime, mag_x, mag_y, mag_z, accel_x, accel_y, accel_z,
    rot_r, rot_p, rot_h (10 columns). Returns a sorted numpy array of utimes.
    """
    utimes = []
    with open(ms25_path, "r") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 10:
                continue
            try:
                utime = float(parts[0])
            except ValueError:
                continue
            utimes.append(utime)
    return np.array(sorted(utimes))


def load_scan_timestamps(sync_dir):
    filenames = [f for f in os.listdir(sync_dir) if f.endswith(".bin")]
    utimes = [int(f.replace(".bin", "")) for f in filenames]
    return np.array(sorted(utimes))


def check_alignment(imu_utimes, scan_utimes, tolerance_us=50000):
    print(f"IMU (ms25): {len(imu_utimes)} timestamps, "
          f"span [{imu_utimes[0]}, {imu_utimes[-1]}]")
    print(f"Velodyne scans: {len(scan_utimes)} timestamps, "
          f"span [{scan_utimes[0]}, {scan_utimes[-1]}]")

    imu_duration_s = (imu_utimes[-1] - imu_utimes[0]) / 1e6
    scan_duration_s = (scan_utimes[-1] - scan_utimes[0]) / 1e6
    print(f"\nIMU duration:      {imu_duration_s:.1f} s")
    print(f"Velodyne duration: {scan_duration_s:.1f} s")

    imu_dt = np.diff(imu_utimes)
    print(f"\nIMU sample interval: mean={imu_dt.mean():.0f} us "
          f"(~{1e6/imu_dt.mean():.1f} Hz), median={np.median(imu_dt):.0f} us")

    overlap_start = max(imu_utimes[0], scan_utimes[0])
    overlap_end = min(imu_utimes[-1], scan_utimes[-1])
    overlap_s = (overlap_end - overlap_start) / 1e6
    print(f"Overlapping time window: {overlap_s:.1f} s")

    gaps = []
    matched = 0
    for st in scan_utimes:
        idx = np.searchsorted(imu_utimes, st)
        candidates = []
        if idx > 0:
            candidates.append(imu_utimes[idx - 1])
        if idx < len(imu_utimes):
            candidates.append(imu_utimes[idx])
        if not candidates:
            continue
        nearest = min(candidates, key=lambda t: abs(t - st))
        gap = abs(nearest - st)
        gaps.append(gap)
        if gap <= tolerance_us:
            matched += 1

    gaps = np.array(gaps)
    print(f"\n--- Match quality (tolerance: {tolerance_us/1000:.0f} ms) ---")
    print(f"Scans with an IMU sample within tolerance: "
          f"{matched} / {len(scan_utimes)} ({100*matched/len(scan_utimes):.1f}%)")
    print(f"Gap to nearest IMU timestamp (microseconds):")
    print(f"  mean={gaps.mean():.0f}  median={np.median(gaps):.0f}  "
          f"max={gaps.max():.0f}  (= {gaps.max()/1000:.1f} ms)")

    bad = np.sum(gaps > tolerance_us)
    print(f"\nScans with NO IMU match within tolerance: {bad} ({100*bad/len(scan_utimes):.1f}%)")
    if bad > 0:
        worst_idx = np.argsort(gaps)[-5:][::-1]
        print("Worst 5 gaps (microseconds):")
        for i in worst_idx:
            print(f"  scan utime={scan_utimes[i]}, gap={gaps[i]:.0f}")

    print("\n--- What this means for IMU-based ICP initialization ---")
    avg_imu_per_scan = (1e6 / imu_dt.mean()) / (1e6 / np.diff(scan_utimes).mean())
    print(f"Roughly {avg_imu_per_scan:.1f} IMU samples available per LiDAR scan interval.")
    print("If this is >= a few samples, IMU-based yaw/velocity integration between")
    print("consecutive scans should have enough resolution to be meaningful.")
    print("If it's close to 1 or less, IMU init will be similar in spirit to")
    print("KITTI's sync data limitation we hit earlier -- worth checking before relying on it.")


def main():
    if len(sys.argv) < 3:
        print("Usage: python3 check_nclt_imu_alignment.py <ms25_csv> <velodyne_sync_dir> [tolerance_ms]")
        sys.exit()

    ms25_path = sys.argv[1]
    sync_dir = sys.argv[2]
    tolerance_ms = float(sys.argv[3]) if len(sys.argv) > 3 else 50.0

    print("Loading IMU (ms25.csv) timestamps...")
    imu_utimes = load_ms25_timestamps(ms25_path)

    print("Loading velodyne_sync timestamps (from filenames)...")
    scan_utimes = load_scan_timestamps(sync_dir)

    check_alignment(imu_utimes, scan_utimes, tolerance_us=tolerance_ms * 1000)


if __name__ == "__main__":
    main()