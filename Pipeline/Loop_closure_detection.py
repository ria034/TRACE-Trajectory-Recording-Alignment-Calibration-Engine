import sys
import os
sys.path.insert(0, r"C:\Users\lenovo\Downloads\Pose_graph_sparsification_research\g2o_file_generation\Pipeline\PyICP-SLAM")
sys.path.insert(0, r"C:\Users\lenovo\Downloads\Pose_graph_sparsification_research\g2o_file_generation\Pipeline\PyICP-SLAM\utils")
import time
import numpy as np
import open3d as o3d
from math import radians, cos, sin
from ScanContextManager import ScanContextManager


def load_velodyne_bin(bin_filename):
    raw = np.fromfile(bin_filename, dtype=np.float32)
    points = raw.reshape(-1, 4)
    return points[:, :-1]


def load_all_scans(lidar_input_file_path):
    all_scans = []
    for filename in sorted(os.listdir(lidar_input_file_path)):
        if filename.endswith(".bin"):
            bin_filename = os.path.join(lidar_input_file_path, filename)
            all_scans.append(load_velodyne_bin(bin_filename))
    return all_scans


def detect_loop_closures(all_scans):
    """
    Returns loop candidates as tuples: (target_idx, source_idx, yaw_diff_deg)
    """
    scm = ScanContextManager(shape=[20, 60], num_candidates=10, threshold=0.15)
    loop_candidates = []

    for idx, scan in enumerate(all_scans):
        scm.addNode(idx, scan)
        nn_idx, nn_dist, nn_yawdiff_deg = scm.detectLoop()
        if nn_idx is not None:
            loop_candidates.append((nn_idx, idx, nn_yawdiff_deg))

    return loop_candidates


def get_initial_yaw_transform(yaw_deg):
    """Creates a 4x4 matrix representing a pure yaw rotation around Z axis."""
    rad = radians(yaw_deg)
    c, s = cos(rad), sin(rad)
    init_T = np.eye(4)
    init_T[0, 0] = c
    init_T[0, 1] = -s
    init_T[1, 0] = s
    init_T[1, 1] = c
    return init_T


def run_icp(points_i, points_j, yaw_diff_deg=0.0):
    """
    Aligns points_j (source) onto points_i (target) using point-to-plane ICP
    seeded with ScanContext's estimated yaw rotation.
    """
    source = o3d.geometry.PointCloud()
    source.points = o3d.utility.Vector3dVector(points_j)
    target = o3d.geometry.PointCloud()
    target.points = o3d.utility.Vector3dVector(points_i)

    # 1. Downsample first for performance and robustness
    source_down = source.voxel_down_sample(voxel_size=0.5)
    target_down = target.voxel_down_sample(voxel_size=0.5)

    # 2. Compute target normals on downsampled cloud
    target_down.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1.0, max_nn=30)
    )

    # 3. Warm-start ICP with ScanContext relative yaw rotation
    init_guess = get_initial_yaw_transform(yaw_diff_deg)

    result = o3d.pipelines.registration.registration_icp(
        source_down,
        target_down,
        max_correspondence_distance=1.0,
        init=init_guess,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50),
    )

    if result.fitness < 0.3:
        return None

    return result.transformation


def rotation_matrix_to_quaternion(R):
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


def write_loop_closures(loop_candidates, all_scans, odometric_file_path):
    # Ensure target path points directly to the file, not the folder
    if os.path.isdir(odometric_file_path):
        odometric_file_path = os.path.join(odometric_file_path, "kitti_0018.g2o")

    # Ensure target output directory exists
    os.makedirs(os.path.dirname(odometric_file_path), exist_ok=True)

    lines_to_write = []
    for scan_i, scan_j, yaw_diff in loop_candidates:
        points_i = all_scans[scan_i]
        points_j = all_scans[scan_j]

        T = run_icp(points_i, points_j, yaw_diff_deg=yaw_diff)
        if T is None:
            print(f"  Skipped ({scan_i}, {scan_j}): ICP fit under threshold (< 0.3)")
            continue

        R = T[:3, :3]
        t = T[:3, 3]
        dx, dy, dz = t[0], t[1], t[2]
        qx, qy, qz, qw = rotation_matrix_to_quaternion(R)

        # Placeholder diagonal information matrix (will be overwritten by step 3)
        info_values = [
            500.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            500.0, 0.0, 0.0, 0.0, 0.0,
            500.0, 0.0, 0.0, 0.0,
            500.0, 0.0, 0.0,
            500.0, 0.0,
            500.0
        ]
        info_str = " ".join(f"{v:.1f}" for v in info_values)

        edge_str = (
            f"EDGE_SE3:QUAT {scan_i} {scan_j} "
            f"{dx:.6f} {dy:.6f} {dz:.6f} {qx:.6f} {qy:.6f} {qz:.6f} {qw:.6f} {info_str}\n"
        )
        lines_to_write.append(edge_str)
        print(f"  Wrote loop edge ({scan_i} -> {scan_j})")

    # Safe write block with retry logic to bypass temporary Windows locks
    max_retries = 5
    for attempt in range(1, max_retries + 1):
        try:
            with open(odometric_file_path, "a", encoding="utf-8") as f:
                f.writelines(lines_to_write)
                f.flush()
                os.fsync(f.fileno())
            print(f"Done. {len(lines_to_write)} loop closure edges appended to {odometric_file_path}")
            break
        except PermissionError as e:
            if attempt == max_retries:
                raise e
            print(f"[WARNING] Output file locked on attempt {attempt}/{max_retries}. Retrying...")
            time.sleep(1.0)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 detect_loop_closures.py <odometric_g2o_file> <lidar_data_dir>")
        sys.exit(1)

    odometric_file_path = sys.argv[1]
    lidar_input_file_path = sys.argv[2]

    print("Loading point clouds...")
    scans = load_all_scans(lidar_input_file_path)
    print(f"Loaded {len(scans)} scans.")

    print("Detecting loop closures with ScanContext...")
    loop_candidates = detect_loop_closures(scans)
    print(f"Found {len(loop_candidates)} loop closure candidates.")

    write_loop_closures(loop_candidates, scans, odometric_file_path)