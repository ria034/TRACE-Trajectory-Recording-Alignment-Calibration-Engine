import sys
import os
import numpy as np


def load_groundtruth_timestamps(gt_path):
    """Returns a sorted numpy array of valid (non-NaN) groundtruth utimes."""
    utimes = []
    with open(gt_path, "r") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 7:
                continue
            try:
                utime = float(parts[0])
                x = float(parts[1])
            except ValueError:
                continue
            if np.isnan(x):
                continue
            utimes.append(utime)
    return np.array(sorted(utimes))


def load_scan_timestamps(sync_dir):
    """Returns a sorted numpy array of utimes derived from velodyne_sync filenames."""
    filenames = [f for f in os.listdir(sync_dir) if f.endswith(".bin")]
    utimes = [int(f.replace(".bin", "")) for f in filenames]
    return np.array(sorted(utimes))


def check_alignment(gt_utimes, scan_utimes, tolerance_us=50000):
    """
    tolerance_us: how close (in microseconds) a scan's timestamp needs to
    be to its nearest groundtruth timestamp to count as "matched".
    Default 50,000 us = 50ms.
    """
    print(f"Groundtruth: {len(gt_utimes)} valid timestamps, "
          f"span [{gt_utimes[0]}, {gt_utimes[-1]}]")
    print(f"Velodyne scans: {len(scan_utimes)} timestamps, "
          f"span [{scan_utimes[0]}, {scan_utimes[-1]}]")

    gt_duration_s = (gt_utimes[-1] - gt_utimes[0]) / 1e6
    scan_duration_s = (scan_utimes[-1] - scan_utimes[0]) / 1e6
    print(f"\nGroundtruth duration: {gt_duration_s:.1f} s")
    print(f"Velodyne duration:    {scan_duration_s:.1f} s")

    overlap_start = max(gt_utimes[0], scan_utimes[0])
    overlap_end = min(gt_utimes[-1], scan_utimes[-1])
    overlap_s = (overlap_end - overlap_start) / 1e6
    print(f"Overlapping time window: {overlap_s:.1f} s")

    # For each scan, find nearest groundtruth timestamp and the gap
    gaps = []
    matched = 0
    for st in scan_utimes:
        idx = np.searchsorted(gt_utimes, st)
        candidates = []
        if idx > 0:
            candidates.append(gt_utimes[idx - 1])
        if idx < len(gt_utimes):
            candidates.append(gt_utimes[idx])
        if not candidates:
            continue
        nearest = min(candidates, key=lambda g: abs(g - st))
        gap = abs(nearest - st)
        gaps.append(gap)
        if gap <= tolerance_us:
            matched += 1

    gaps = np.array(gaps)
    print(f"\n--- Match quality (tolerance: {tolerance_us/1000:.0f} ms) ---")
    print(f"Scans with a groundtruth match within tolerance: "
          f"{matched} / {len(scan_utimes)} ({100*matched/len(scan_utimes):.1f}%)")
    print(f"Gap to nearest groundtruth timestamp (microseconds):")
    print(f"  mean={gaps.mean():.0f}  median={np.median(gaps):.0f}  "
          f"max={gaps.max():.0f}  (= {gaps.max()/1000:.1f} ms)")

    # Flag scans with unusually large gaps (outside overlap window, or
    # genuine groundtruth dropouts)
    bad = np.sum(gaps > tolerance_us)
    print(f"\nScans with NO match within tolerance: {bad} ({100*bad/len(scan_utimes):.1f}%)")
    if bad > 0:
        worst_idx = np.argsort(gaps)[-5:][::-1]
        print("Worst 5 gaps (microseconds):")
        for i in worst_idx:
            print(f"  scan utime={scan_utimes[i]}, gap={gaps[i]:.0f}")


def main():
    if len(sys.argv) < 3:
        print("Usage: python3 check_nclt_timestamp_alignment.py <groundtruth_csv> <velodyne_sync_dir> [tolerance_ms]")
        sys.exit()

    gt_path = sys.argv[1]
    sync_dir = sys.argv[2]
    tolerance_ms = float(sys.argv[3]) if len(sys.argv) > 3 else 50.0

    print("Loading groundtruth timestamps...")
    gt_utimes = load_groundtruth_timestamps(gt_path)

    print("Loading velodyne_sync timestamps (from filenames, no need to read file contents)...")
    scan_utimes = load_scan_timestamps(sync_dir)

    check_alignment(gt_utimes, scan_utimes, tolerance_us=tolerance_ms * 1000)


if __name__ == "__main__":
    main()