import sys
import os
import numpy as np


def load_cov_timestamps(cov_path):
    """Returns a sorted numpy array of utimes from cov_2012-01-08.csv --
    these mark the genuinely precise SLAM graph node timestamps (roughly
    every 8 meters traveled), as opposed to the much denser but mostly
    odometry-interpolated groundtruth file."""
    utimes = []
    with open(cov_path, "r") as f:
        for line in f:
            parts = line.strip().split(",")
            if not parts:
                continue
            try:
                utimes.append(int(float(parts[0])))
            except ValueError:
                continue
    return np.array(sorted(utimes))


def load_groundtruth_filtered(gt_path, precise_utimes, tolerance_us=50000):
    """
    Loads ALL groundtruth rows, then for each PRECISE SLAM-node timestamp
    (from cov_2012-01-08.csv), finds the NEAREST groundtruth row in time
    -- not requiring exact utime equality, since cov and groundtruth
    timestamps aren't necessarily bit-identical even when referring to
    the same underlying node (confirmed: NCLT's own official
    read_ground_truth.py uses scipy.interpolate.interp1d(kind='nearest')
    for exactly this reason, rather than exact matching).

    Keeps z in its NATIVE form (NED, positive=down) -- confirmed against
    NCLT's own official read_ground_truth.py reference script, which only
    negates z for PLOT display purposes (colorbar intuition), never for
    the actual data used in computation. Since our Velodyne point-cloud
    reader also applies no sign convention change, keeping z untouched
    here maintains consistency with whatever native frame convention the
    LiDAR sensor calibration already uses.

    Returns (utimes, poses) where poses[i] = (x, y, z, roll, pitch, yaw).
    """
    all_rows = []
    with open(gt_path, "r") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 7:
                continue
            try:
                vals = [float(v) for v in parts[:7]]
            except ValueError:
                continue
            if any(np.isnan(v) for v in vals[1:]):
                continue
            all_rows.append(vals)

    all_rows.sort(key=lambda r: r[0])
    all_gt_utimes = np.array([r[0] for r in all_rows])

    matched_rows = []
    dropped = 0
    for putime in precise_utimes:
        idx = np.searchsorted(all_gt_utimes, putime)
        candidates = []
        if idx > 0:
            candidates.append(idx - 1)
        if idx < len(all_gt_utimes):
            candidates.append(idx)
        if not candidates:
            dropped += 1
            continue

        best_idx = min(candidates, key=lambda i: abs(all_gt_utimes[i] - putime))
        gap = abs(all_gt_utimes[best_idx] - putime)

        if gap > tolerance_us:
            dropped += 1
            continue

        matched_rows.append(all_rows[best_idx])

    if dropped > 0:
        print(f"  (dropped {dropped} / {len(precise_utimes)} precise timestamps "
              f"with no groundtruth match within {tolerance_us/1000:.0f}ms)")

    matched_rows.sort(key=lambda r: r[0])
    utimes = np.array([r[0] for r in matched_rows])
    poses = [tuple(r[1:]) for r in matched_rows]
    return utimes, poses


def load_scan_utimes(sync_dir):
    filenames = sorted(f for f in os.listdir(sync_dir) if f.endswith(".bin"))
    utimes = [int(f.replace(".bin", "")) for f in filenames]
    return filenames, np.array(utimes)


def match_precise_poses_to_scans(precise_utimes, precise_poses, scan_filenames,
                                  scan_utimes, tolerance_us=100000):
    """
    For each PRECISE groundtruth pose (the sparse, ~1300-point set), finds
    its nearest LiDAR scan. This is the REVERSE matching direction from
    the dense-groundtruth version -- here there are far fewer precise
    points than scans, so we match each precise point to its closest scan,
    not the other way around.
    """
    matched = []
    dropped = 0

    for putime, ppose in zip(precise_utimes, precise_poses):
        idx = np.searchsorted(scan_utimes, putime)
        candidates = []
        if idx > 0:
            candidates.append(idx - 1)
        if idx < len(scan_utimes):
            candidates.append(idx)
        if not candidates:
            dropped += 1
            continue

        best_idx = min(candidates, key=lambda i: abs(scan_utimes[i] - putime))
        gap = abs(scan_utimes[best_idx] - putime)

        if gap > tolerance_us:
            dropped += 1
            continue

        matched.append((scan_filenames[best_idx], scan_utimes[best_idx], ppose))

    print(f"Matched {len(matched)} / {len(precise_utimes)} precise groundtruth "
          f"poses to scans (dropped {dropped} with no scan within "
          f"{tolerance_us/1000:.0f}ms)")
    return matched


def convert_angles_to_quat(roll, pitch, yaw):
    cr, sr = np.cos(roll / 2.0), np.sin(roll / 2.0)
    cp, sp = np.cos(pitch / 2.0), np.sin(pitch / 2.0)
    cy, sy = np.cos(yaw / 2.0), np.sin(yaw / 2.0)
    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    return qw, qx, qy, qz


def quat_to_rotation_matrix(qx, qy, qz, qw):
    n = np.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    qx, qy, qz, qw = qx / n, qy / n, qz / n, qw / n
    return np.array([
        [1 - 2 * (qy ** 2 + qz ** 2), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx ** 2 + qz ** 2), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx ** 2 + qy ** 2)],
    ])


def rotation_matrix_to_quat(R):
    trace = np.trace(R)
    if trace > 0:
        S = np.sqrt(trace + 1.0) * 2
        qw = 0.25 * S
        qx = (R[2, 1] - R[1, 2]) / S
        qy = (R[0, 2] - R[2, 0]) / S
        qz = (R[1, 0] - R[0, 1]) / S
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        qw = (R[2, 1] - R[1, 2]) / S
        qx = 0.25 * S
        qy = (R[0, 1] + R[1, 0]) / S
        qz = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        qw = (R[0, 2] - R[2, 0]) / S
        qx = (R[0, 1] + R[1, 0]) / S
        qy = 0.25 * S
        qz = (R[1, 2] + R[2, 1]) / S
    else:
        S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        qw = (R[1, 0] - R[0, 1]) / S
        qx = (R[0, 2] + R[2, 0]) / S
        qy = (R[1, 2] + R[2, 1]) / S
        qz = 0.25 * S
    return qx, qy, qz, qw


def compute_relative_transform(pose_a, pose_b):
    xa, ya, za, qwa, qxa, qya, qza = pose_a
    xb, yb, zb, qwb, qxb, qyb, qzb = pose_b
    Ra = quat_to_rotation_matrix(qxa, qya, qza, qwa)
    Rb = quat_to_rotation_matrix(qxb, qyb, qzb, qwb)
    ta = np.array([xa, ya, za])
    tb = np.array([xb, yb, zb])
    R_rel = Ra.T @ Rb
    t_rel = Ra.T @ (tb - ta)
    qx, qy, qz, qw = rotation_matrix_to_quat(R_rel)
    return t_rel[0], t_rel[1], t_rel[2], qx, qy, qz, qw


def main():
    if len(sys.argv) < 5:
        print("Usage: python3 sync_nclt_precise_scans.py "
              "<groundtruth_csv> <cov_csv> <velodyne_sync_dir> <output_dir>")
        sys.exit()

    gt_path = sys.argv[1]
    cov_path = sys.argv[2]
    sync_dir = sys.argv[3]
    output_dir = sys.argv[4]
    os.makedirs(output_dir, exist_ok=True)

    print("Loading precise SLAM-node timestamps from cov file...")
    precise_utimes = load_cov_timestamps(cov_path)
    print(f"Found {len(precise_utimes)} precise timestamps")

    print("Loading and filtering groundtruth to precise timestamps only...")
    gt_utimes_filtered, gt_poses_filtered = load_groundtruth_filtered(gt_path, precise_utimes)
    print(f"Matched {len(gt_utimes_filtered)} groundtruth rows to precise timestamps")

    print("Loading scan timestamps...")
    scan_filenames, scan_utimes = load_scan_utimes(sync_dir)
    print(f"Found {len(scan_filenames)} scans")

    matched = match_precise_poses_to_scans(
        gt_utimes_filtered, gt_poses_filtered, scan_filenames, scan_utimes)

    scan_order_path = os.path.join(output_dir, "scan_order_precise.txt")
    with open(scan_order_path, "w") as f:
        for filename, utime, pose in matched:
            f.write(f"{filename}\n")
    print(f"Wrote scan order to {scan_order_path}")

    all_poses = []
    for filename, utime, (x, y, z, roll, pitch, yaw) in matched:
        qw, qx, qy, qz = convert_angles_to_quat(roll, pitch, yaw)
        all_poses.append((x, y, z, qw, qx, qy, qz))

    output_file = os.path.join(output_dir, "nclt_trajectory_precise.g2o")
    with open(output_file, "w") as f:
        for vid, (x, y, z, qw, qx, qy, qz) in enumerate(all_poses):
            f.write(f"VERTEX_SE3:QUAT {vid} {x} {y} {z} {qx} {qy} {qz} {qw}\n")

        for vid in range(len(all_poses) - 1):
            dx, dy, dz, qx, qy, qz, qw = compute_relative_transform(
                all_poses[vid], all_poses[vid + 1])
            f.write(f"EDGE_SE3:QUAT {vid} {vid + 1} {dx} {dy} {dz} {qx} {qy} {qz} {qw}\n")

    print(f"\nWrote {len(all_poses)} vertices and {len(all_poses)-1} edges to {output_file}")


if __name__ == "__main__":
    main()