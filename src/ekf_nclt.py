import sys
import os
import numpy as np
import open3d as o3d


# ---------------------------------------------------------------------------
# Scan loading -- directly from the velodyne_sync folder, no scan_order.txt
# ---------------------------------------------------------------------------

HIT_DTYPE_SIZE = 8  # 3x uint16 (x,y,z) + 2x uint8 (intensity, ring)
SCALE = 0.005
OFFSET = -100.0


def load_synced_scan(filepath):
    with open(filepath, "rb") as f:
        data = f.read()
    n_hits = len(data) // HIT_DTYPE_SIZE
    import struct
    hit_struct = struct.Struct("<HHHBB")
    hits = np.zeros((n_hits, 3))
    for i in range(n_hits):
        chunk = data[i*HIT_DTYPE_SIZE:(i+1)*HIT_DTYPE_SIZE]
        x_raw, y_raw, z_raw, intensity, ring = hit_struct.unpack(chunk)
        hits[i, 0] = x_raw * SCALE + OFFSET
        hits[i, 1] = y_raw * SCALE + OFFSET
        hits[i, 2] = z_raw * SCALE + OFFSET
    return hits


class DirectScanLoader:
    """Lists scans directly from velodyne_sync, sorted by filename (=utime).
    No separate scan_order.txt needed -- always uses every scan in the folder."""

    def __init__(self, sync_dir):
        self.sync_dir = sync_dir
        self.filenames = sorted(f for f in os.listdir(sync_dir) if f.endswith(".bin"))

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        filepath = os.path.join(self.sync_dir, self.filenames[index])
        return load_synced_scan(filepath)


# ---------------------------------------------------------------------------
# Rotation utilities
# ---------------------------------------------------------------------------

def skew(v):
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0],
    ])


def rotation_vector_to_matrix(rv):
    theta = np.linalg.norm(rv)
    if theta < 1e-12:
        return np.eye(3)
    axis = rv / theta
    K = skew(axis)
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)


def rotation_matrix_to_rotvec(R):
    trace = np.clip((np.trace(R) - 1) / 2, -1.0, 1.0)
    angle = np.arccos(trace)
    if angle < 1e-8:
        return np.zeros(3)
    return angle / (2 * np.sin(angle)) * np.array([
        R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])


def rpy_to_rotation_matrix(roll, pitch, yaw):
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    return Rz @ Ry @ Rx


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


def quat_to_rotation_matrix(qx, qy, qz, qw):
    n = np.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    qx, qy, qz, qw = qx / n, qy / n, qz / n, qw / n
    return np.array([
        [1 - 2 * (qy ** 2 + qz ** 2), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx ** 2 + qz ** 2), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx ** 2 + qy ** 2)],
    ])


def load_checkpoint(checkpoint_path):
    """
    Reads an existing checkpoint (or completed output) g2o file and
    reconstructs the list of (p, R) poses in vertex-id order, so a run
    can resume exactly where a previous one left off instead of starting
    over from scratch.
    """
    verts = {}
    with open(checkpoint_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if not parts or parts[0] != "VERTEX_SE3:QUAT":
                continue
            vid = int(parts[1])
            x, y, z = float(parts[2]), float(parts[3]), float(parts[4])
            qx, qy, qz, qw = float(parts[5]), float(parts[6]), float(parts[7]), float(parts[8])
            R = quat_to_rotation_matrix(qx, qy, qz, qw)
            verts[vid] = (np.array([x, y, z]), R)

    ids = sorted(verts.keys())
    poses_out = [verts[i] for i in ids]
    return poses_out


def compute_imu_to_velo_calib():
    VEL_TRANS = np.array([0.002, -0.004, -0.957])
    VEL_RPY_DEG = np.array([0.807, 0.166, -90.703])
    IMU_TRANS = np.array([-0.11, -0.18, -0.71])
    IMU_RPY_DEG = np.array([0.0, 0.0, 0.0])

    T_body_vel = np.eye(4)
    T_body_vel[:3, :3] = rpy_to_rotation_matrix(*np.radians(VEL_RPY_DEG))
    T_body_vel[:3, 3] = VEL_TRANS

    T_body_imu = np.eye(4)
    T_body_imu[:3, :3] = rpy_to_rotation_matrix(*np.radians(IMU_RPY_DEG))
    T_body_imu[:3, 3] = IMU_TRANS

    return np.linalg.inv(T_body_vel) @ T_body_imu


def transform_delta_to_velo_frame(T_delta, C):
    return C @ T_delta @ np.linalg.inv(C)


# ---------------------------------------------------------------------------
# EKF state, propagate, absolute update
# ---------------------------------------------------------------------------

class EKFState:
    def __init__(self):
        self.p = np.zeros(3)
        self.R = np.eye(3)
        self.P = np.eye(6) * 1e-8


def apply_step(state, dT, confident, trans_noise_scale=0.02, rot_noise_scale=0.01,
                icp_confidence_scale=0.1):
    state.p = state.p + state.R @ dT[:3, 3]
    state.R = state.R @ dT[:3, :3]

    trans_mag = np.linalg.norm(dT[:3, 3])
    rot_mag = np.linalg.norm(rotation_matrix_to_rotvec(dT[:3, :3]))

    scale = icp_confidence_scale if confident else 1.0
    Q = np.zeros((6, 6))
    Q[0:3, 0:3] = np.eye(3) * (scale * trans_noise_scale * max(trans_mag, 0.01)) ** 2
    Q[3:6, 3:6] = np.eye(3) * (scale * rot_noise_scale * max(rot_mag, 0.001)) ** 2

    state.P = state.P + Q
    return state


def update_absolute(state, measured_p, measured_R, measurement_noise=0.02):
    H = np.eye(6)
    R_meas = np.eye(6) * measurement_noise ** 2

    pos_residual = measured_p - state.p
    R_err = state.R.T @ measured_R
    rot_residual = rotation_matrix_to_rotvec(R_err)
    residual = np.concatenate([pos_residual, rot_residual])

    S = H @ state.P @ H.T + R_meas
    try:
        K = state.P @ H.T @ np.linalg.inv(S)
    except np.linalg.LinAlgError:
        return state

    dx = K @ residual
    state.p = state.p + dx[0:3]
    state.R = state.R @ rotation_vector_to_matrix(dx[3:6])
    state.P = (np.eye(6) - K @ H) @ state.P
    return state


# ---------------------------------------------------------------------------
# ICP
# ---------------------------------------------------------------------------

def run_icp(points_i, points_j, T_init, voxel_size=0.5, max_correspondence_distance=1.0):
    source = o3d.geometry.PointCloud()
    source.points = o3d.utility.Vector3dVector(points_j)
    target = o3d.geometry.PointCloud()
    target.points = o3d.utility.Vector3dVector(points_i)

    source_down = source.voxel_down_sample(voxel_size)
    target_down = target.voxel_down_sample(voxel_size)
    target_down.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1.0, max_nn=30))

    result = o3d.pipelines.registration.registration_icp(
        source_down, target_down,
        max_correspondence_distance,
        T_init,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50)
    )
    return result.transformation, result.fitness


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_odometry_100hz(filepath):
    rows = []
    with open(filepath, "r") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 7:
                continue
            try:
                vals = [float(v) for v in parts[:7]]
            except ValueError:
                continue
            rows.append(vals)
    rows.sort(key=lambda r: r[0])
    utimes = np.array([r[0] for r in rows])
    poses = [tuple(r[1:]) for r in rows]
    return utimes, poses


def load_cov_timestamps(cov_path):
    utimes = []
    with open(cov_path, "r") as f:
        for line in f:
            parts = line.strip().split(",")
            try:
                utimes.append(int(float(parts[0])))
            except (ValueError, IndexError):
                continue
    return np.array(sorted(utimes))


def load_groundtruth_at(gt_path, target_utimes, tolerance_us=50000):
    rows = []
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
            rows.append(vals)
    rows.sort(key=lambda r: r[0])
    gt_utimes = np.array([r[0] for r in rows])

    result = {}
    for tu in target_utimes:
        idx = np.searchsorted(gt_utimes, tu)
        candidates = []
        if idx > 0:
            candidates.append(idx - 1)
        if idx < len(gt_utimes):
            candidates.append(idx)
        if not candidates:
            continue
        best = min(candidates, key=lambda i: abs(gt_utimes[i] - tu))
        if abs(gt_utimes[best] - tu) > tolerance_us:
            continue
        result[tu] = tuple(rows[best][1:])
    return result


def get_odom_pose_near(odom_utimes, odom_poses, target_utime):
    idx = np.searchsorted(odom_utimes, target_utime)
    candidates = []
    if idx > 0:
        candidates.append(idx - 1)
    if idx < len(odom_utimes):
        candidates.append(idx)
    best = min(candidates, key=lambda i: abs(odom_utimes[i] - target_utime))
    x, y, z, roll, pitch, yaw = odom_poses[best]
    return np.array([x, y, z]), rpy_to_rotation_matrix(roll, pitch, yaw)


# ---------------------------------------------------------------------------
# Main EKF run
# ---------------------------------------------------------------------------

def write_g2o(poses_out, output_path):
    with open(output_path, "w") as f:
        for vid, (p, R) in enumerate(poses_out):
            qx, qy, qz, qw = rotation_matrix_to_quat(R)
            f.write(f"VERTEX_SE3:QUAT {vid} {p[0]} {p[1]} {p[2]} {qx} {qy} {qz} {qw}\n")

        for i in range(len(poses_out) - 1):
            p1, R1 = poses_out[i]
            p2, R2 = poses_out[i + 1]
            R_rel = R1.T @ R2
            t_rel = R1.T @ (p2 - p1)
            qx, qy, qz, qw = rotation_matrix_to_quat(R_rel)
            f.write(f"EDGE_SE3:QUAT {i} {i+1} {t_rel[0]} {t_rel[1]} {t_rel[2]} "
                    f"{qx} {qy} {qz} {qw}\n")


def build_ekf_trajectory(scan_loader, odom_utimes, odom_poses, precise_gt,
                          C_imu_to_velo, output_path, min_icp_fitness=0.3,
                          checkpoint_every=500, resume_poses=None):
    n = len(scan_loader)

    if resume_poses is not None and len(resume_poses) > 1:
        poses_out = list(resume_poses)
        start_i = len(poses_out) - 1
        state = EKFState()
        state.p, state.R = poses_out[-1]
        print(f"RESUMING from checkpoint: {len(poses_out)} vertices already "
              f"computed, continuing from scan index {start_i}")
    else:
        state = EKFState()
        poses_out = [(state.p.copy(), state.R.copy())]
        start_i = 0

    icp_accepted = 0
    icp_rejected = 0
    corrections_applied = 0

    utime_i0 = int(scan_loader.filenames[start_i].replace(".bin", ""))
    prev_odom_p, prev_odom_R = get_odom_pose_near(odom_utimes, odom_poses, utime_i0)

    for i in range(start_i, n - 1):
        utime_j = int(scan_loader.filenames[i + 1].replace(".bin", ""))

        odom_p_i, odom_R_i = prev_odom_p, prev_odom_R
        odom_p_j, odom_R_j = get_odom_pose_near(odom_utimes, odom_poses, utime_j)

        R_rel = odom_R_i.T @ odom_R_j
        t_rel = odom_R_i.T @ (odom_p_j - odom_p_i)
        T_odom = np.eye(4)
        T_odom[:3, :3] = R_rel
        T_odom[:3, 3] = t_rel

        T_init = transform_delta_to_velo_frame(T_odom, C_imu_to_velo)

        points_i = scan_loader[i]
        points_j = scan_loader[i + 1]
        T_icp, fitness = run_icp(points_i, points_j, T_init)

        if fitness >= min_icp_fitness:
            T_step = T_icp
            confident = True
            icp_accepted += 1
        else:
            T_step = T_init
            confident = False
            icp_rejected += 1

        state = apply_step(state, T_step, confident)

        if utime_j in precise_gt:
            gx, gy, gz, groll, gpitch, gyaw = precise_gt[utime_j]
            measured_R = rpy_to_rotation_matrix(groll, gpitch, gyaw)
            measured_p = np.array([gx, gy, gz])
            state = update_absolute(state, measured_p, measured_R)
            corrections_applied += 1

        poses_out.append((state.p.copy(), state.R.copy()))
        prev_odom_p, prev_odom_R = odom_p_j, odom_R_j

        if (i + 1) % checkpoint_every == 0:
            print(f"  Processed {i+1}/{n-1}: ICP accepted={icp_accepted}, "
                  f"rejected={icp_rejected}, corrections={corrections_applied}, "
                  f"position={state.p}")
            # CHECKPOINT: save progress to disk NOW, so a crash/error later
            # (path typo, out-of-memory, power loss, anything) never loses
            # more than `checkpoint_every` scans worth of computation.
            checkpoint_path = output_path + f".checkpoint"
            try:
                write_g2o(poses_out, checkpoint_path)
                print(f"    (checkpoint saved to {checkpoint_path})")
            except Exception as e:
                print(f"    WARNING: checkpoint save failed: {e}")
                print(f"    (continuing anyway -- but fix the output path NOW "
                      f"before more time is wasted)")

    processed_this_run = (n - 1) - start_i
    print(f"\nICP accepted: {icp_accepted} / {processed_this_run} (this run)")
    print(f"ICP rejected (fallback to odometry): {icp_rejected} / {processed_this_run} (this run)")
    print(f"Absolute corrections applied: {corrections_applied} (this run)")
    return poses_out


def main():
    if len(sys.argv) < 5:
        print("Usage: python3 build_ekf_lidar_trajectory_nclt.py "
              "<odometry_mu_100hz.csv> <groundtruth_csv> <cov_csv> "
              "<velodyne_sync_dir> [output_dir]")
        print("  No scan_order.txt needed -- reads every scan directly from")
        print("  velodyne_sync_dir, sorted by filename.")
        print("  output_dir is a FOLDER (created if it doesn't exist) --")
        print("  the actual output file will be output_dir/nclt_ekf_lidar_trajectory.g2o")
        sys.exit()

    odom_path = sys.argv[1]
    gt_path = sys.argv[2]
    cov_path = sys.argv[3]
    sync_dir = sys.argv[4]
    output_dir = sys.argv[5] if len(sys.argv) > 5 else "nclt_ekf_output"

    # The last argument is ALWAYS a folder now -- no more ambiguity between
    # "did I pass a file or a directory" (this is exactly the mistake that
    # cost 3 hours of computation earlier). We create the folder if needed,
    # and write fixed, predictable filenames inside it.
    if os.path.isfile(output_dir):
        print(f"ERROR: '{output_dir}' is an existing FILE, but this argument "
              f"must be a FOLDER (it will be created if it doesn't exist).")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "nclt_ekf_lidar_trajectory.g2o")

    # Validate the output path is actually writable BEFORE running any
    # computation -- this script can take HOURS; failing only at the very
    # end would be an unacceptable, avoidable waste of time.
    try:
        with open(output_path, "a"):
            pass  # touch-test: confirm we can actually write here
    except (PermissionError, OSError) as e:
        print(f"ERROR: cannot write to '{output_path}': {e}")
        print(f"Fix this BEFORE running -- this script can take hours.")
        sys.exit(1)
    print(f"Output folder ready: {output_dir}")
    print(f"Final output will be written to: {output_path}")
    print(f"Checkpoints will be written to:  {output_path}.checkpoint")

    print("Setting up scan loader (direct from folder, no scan_order.txt)...")
    scans = DirectScanLoader(sync_dir)
    print(f"Scan loader ready: {len(scans)} scans")

    print("Loading 100Hz odometry...")
    odom_utimes, odom_poses = load_odometry_100hz(odom_path)
    print(f"Loaded {len(odom_utimes)} odometry samples")

    print("Loading precise SLAM-node checkpoint timestamps...")
    precise_utimes = load_cov_timestamps(cov_path)
    print(f"Found {len(precise_utimes)} precise timestamps")

    print("Matching precise checkpoints to groundtruth poses...")
    precise_gt = load_groundtruth_at(gt_path, precise_utimes)
    print(f"Matched {len(precise_gt)} precise groundtruth poses")

    print("Computing IMU-to-Velodyne calibration...")
    C_imu_to_velo = compute_imu_to_velo_calib()

    checkpoint_path = output_path + ".checkpoint"
    resume_poses = None
    if os.path.exists(checkpoint_path):
        print(f"\nFound existing checkpoint: {checkpoint_path}")
        resume_poses = load_checkpoint(checkpoint_path)
        print(f"Loaded {len(resume_poses)} previously-computed vertices -- "
              f"will resume from there instead of starting over.")
    elif os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        # In case a previous run finished writing the FINAL file but you
        # want to extend/re-verify it, also allow resuming from a
        # completed output (harmless if it's already the full session --
        # the loop will simply find nothing left to process).
        print(f"\nFound existing output file: {output_path}")
        existing = load_checkpoint(output_path)
        if len(existing) > 1:
            print(f"Loaded {len(existing)} previously-computed vertices -- "
                  f"will resume from there instead of starting over.")
            resume_poses = existing

    print("Running EKF (100Hz odometry propagation + ICP step refinement + "
          "absolute corrections)...")
    print("(Progress + checkpoints will be saved every 500 scans -- if this")
    print(" crashes partway through, just run the SAME command again; it")
    print(" will automatically resume from the last checkpoint.)")
    poses_out = build_ekf_trajectory(scans, odom_utimes, odom_poses, precise_gt,
                                      C_imu_to_velo, output_path,
                                      resume_poses=resume_poses)

    print(f"Writing final {len(poses_out)} vertices to {output_path}...")
    write_g2o(poses_out, output_path)
    print(f"Done. Wrote {output_path}")


if __name__ == "__main__":
    main()