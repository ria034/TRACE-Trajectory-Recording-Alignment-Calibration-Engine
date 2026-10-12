import sys
sys.path.insert(0, r"C:\Users\lenovo\Downloads\Pose_graph_sparsification_research\g2o_file_generation\Pipeline\PyICP-SLAM")
sys.path.insert(0, r"C:\Users\lenovo\Downloads\Pose_graph_sparsification_research\g2o_file_generation\Pipeline\PyICP-SLAM\utils")
import os
import glob
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree


def load_velodyne_bin(bin_filename):
    raw = np.fromfile(bin_filename, dtype=np.float32)
    points = raw.reshape(-1, 4)
    return points[:, :-1]


def load_all_scans(lidar_input_file_path):
    all_scans = []
    for filename in sorted(os.listdir(lidar_input_file_path)):
        bin_filename = os.path.join(lidar_input_file_path, filename)
        all_scans.append(load_velodyne_bin(bin_filename))
    return all_scans


def load_oxts_raw(oxts_dir):
    files = sorted(glob.glob(os.path.join(oxts_dir, "*.txt")))
    raw = []
    for f in files:
        with open(f, "r") as fh:
            values = [float(v) for v in fh.readline().strip().split()]
        raw.append(values)
    return raw


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


def rotation_vector_to_matrix(rv):
    theta = np.linalg.norm(rv)
    if theta < 1e-12:
        return np.eye(3)
    axis = rv / theta
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0],
    ])
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)


def load_imu_to_velo_calib(filepath):
    """
    Parses KITTI's calib_imu_to_velo.txt format:
        R: r11 r12 r13 r21 r22 r23 r31 r32 r33
        T: t1 t2 t3
    Returns a 4x4 transform C such that: p_velo = C @ p_imu
    (maps a point expressed in IMU-frame coordinates into Velodyne-frame
    coordinates).
    """
    R = None
    T = None
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("R:"):
                vals = [float(v) for v in line[2:].strip().split()]
                R = np.array(vals).reshape(3, 3)
            elif line.startswith("T:"):
                vals = [float(v) for v in line[2:].strip().split()]
                T = np.array(vals)

    if R is None or T is None:
        raise ValueError(f"Could not parse R and T from {filepath}")

    C = np.eye(4)
    C[:3, :3] = R
    C[:3, 3] = T
    return C


def transform_delta_to_velo_frame(T_imu_delta, C):
    """
    Converts a relative-motion transform expressed in IMU-frame
    conventions into the equivalent relative motion expressed in
    Velodyne-frame conventions, using the fixed rigid calibration C
    (p_velo = C @ p_imu).

    This is the standard "change of reference frame for a relative
    transform between two rigidly-attached sensors" formula:
        T_velo_delta = C @ T_imu_delta @ C^-1
    """
    return C @ T_imu_delta @ np.linalg.inv(C)


def imu_delta_pose(oxts_values, dt=0.1):
    """
    IMU-only (body-frame velocity + angular rate) initial guess for ICP on
    CONSECUTIVE odometry edges -- deliberately avoids using absolute GPS
    position, so ICP has genuine room to correct real drift.
    T maps a point in frame (i+1)'s local coords into frame i's local
    coords, consistent with how compute_relative_transform and ICP's
    source->target convention are used elsewhere in this pipeline.
    """
    vf, vl, vu = oxts_values[8], oxts_values[9], oxts_values[10]
    wf, wl, wu = oxts_values[20], oxts_values[21], oxts_values[22]

    delta_rotation_vec = np.array([wf, wl, wu]) * dt
    R_delta = rotation_vector_to_matrix(delta_rotation_vec)
    delta_position_body = np.array([vf, vl, vu]) * dt

    T = np.eye(4)
    T[:3, :3] = R_delta
    T[:3, 3] = delta_position_body
    return T


def skew(v):
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0],
    ])


def parse_g2o_file(filepath):
    vertices = {}
    edges = []

    with open(filepath, "r") as f:
        for line in f:
            stripped = line.strip()
            if stripped.startswith("VERTEX_SE3:QUAT"):
                parts = stripped.split()
                vid = int(parts[1])
                x, y, z = float(parts[2]), float(parts[3]), float(parts[4])
                qx, qy, qz, qw = float(parts[5]), float(parts[6]), float(parts[7]), float(parts[8])
                vertices[vid] = (x, y, z, qw, qx, qy, qz)
            elif stripped.startswith("EDGE_SE3:QUAT"):
                parts = stripped.split()
                id1 = int(parts[1])
                id2 = int(parts[2])
                dx, dy, dz = float(parts[3]), float(parts[4]), float(parts[5])
                qx, qy, qz, qw = float(parts[6]), float(parts[7]), float(parts[8]), float(parts[9])
                edges.append({
                    "id1": id1, "id2": id2,
                    "dx": dx, "dy": dy, "dz": dz,
                    "qx": qx, "qy": qy, "qz": qz, "qw": qw,
                })

    return vertices, edges


def compute_relative_transform(pose_a, pose_b):
    """
    Returns the transform T_ab such that: p_a = T_ab @ p_b
    i.e. maps a point expressed in b's local frame into a's local frame.
    This matches ICP's source(b)->target(a) convention used throughout
    this pipeline.
    """
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


def ensure_odometry_edges(vertices, edges):
    existing_pairs = set()
    for e in edges:
        existing_pairs.add((e["id1"], e["id2"]))
        existing_pairs.add((e["id2"], e["id1"]))

    max_id = max(vertices.keys())
    added = 0
    for vid in range(max_id):
        if vid not in vertices or (vid + 1) not in vertices:
            continue
        if (vid, vid + 1) in existing_pairs:
            continue
        dx, dy, dz, qx, qy, qz, qw = compute_relative_transform(
            vertices[vid], vertices[vid + 1])
        edges.append({
            "id1": vid, "id2": vid + 1,
            "dx": dx, "dy": dy, "dz": dz,
            "qx": qx, "qy": qy, "qz": qz, "qw": qw,
        })
        added += 1

    print(f"Added {added} missing odometry edges.")
    return edges


def run_icp_refine(points_i, points_j, T_init, voxel_size=0.5,
                    max_correspondence_distance=1.0):
    """
    T aligns SOURCE (points_j, id2's scan) onto TARGET (points_i, id1's
    scan) -- i.e. maps points_j's local frame into points_i's local frame.
    """
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


def estimate_point_to_plane_covariance(points_i, points_j, T,
                                        voxel_size=0.5,
                                        correspondence_dist_threshold=1.0,
                                        min_correspondences=20,
                                        max_correspondences=300):
    """
    Downsamples both clouds AND caps the number of correspondences used.
    Both steps prevent tens of thousands of redundant, spatially-correlated
    raw LiDAR points from being treated as independent measurements, which
    massively (by orders of magnitude) overstates confidence in the
    resulting information matrix if left unchecked.
    """
    pc_i = o3d.geometry.PointCloud()
    pc_i.points = o3d.utility.Vector3dVector(points_i)
    pc_i = pc_i.voxel_down_sample(voxel_size)
    points_i_ds = np.asarray(pc_i.points)

    pc_j = o3d.geometry.PointCloud()
    pc_j.points = o3d.utility.Vector3dVector(points_j)
    pc_j = pc_j.voxel_down_sample(voxel_size)
    points_j_ds = np.asarray(pc_j.points)

    pc_i.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1.0, max_nn=30))
    target_normals = np.asarray(pc_i.normals)

    R = T[:3, :3]
    t = T[:3, 3]
    transformed = (R @ points_j_ds.T).T + t

    tree = cKDTree(points_i_ds)
    dists, idxs = tree.query(transformed, k=1)

    valid = np.where(dists < correspondence_dist_threshold)[0]
    if len(valid) < min_correspondences:
        return None

    if len(valid) > max_correspondences:
        valid = np.random.choice(valid, max_correspondences, replace=False)

    tgt_valid = points_i_ds[idxs[valid]]
    normals_valid = target_normals[idxs[valid]]
    transformed_valid = transformed[valid]

    n = tgt_valid.shape[0]
    residuals = np.zeros(n)
    J = np.zeros((n, 6))

    for i in range(n):
        p_t = transformed_valid[i]
        q = tgt_valid[i]
        normal = normals_valid[i]
        norm_len = np.linalg.norm(normal)
        if norm_len < 1e-9:
            continue
        normal = normal / norm_len

        residuals[i] = normal @ (q - p_t)
        J[i, 0:3] = -normal
        J[i, 3:6] = normal @ skew(p_t)

    JTJ = J.T @ J
    sigma2 = np.sum(residuals ** 2) / max(n - 6, 1)

    # Translation (0:3) and rotation (3:6) blocks of JTJ can differ by
    # orders of magnitude -- rotation columns scale with point RANGE
    # (skew(p_t), often 5-60m for KITTI), while translation columns are
    # just unit normals (~1). A single global damping value calibrated
    # from the overall trace ends up badly mismatched for one block or
    # the other (verified: rotation block trace can be ~1000x the
    # translation block's, making one global damping value give <0.1%
    # effective regularization to whichever block dominates the trace).
    # Fix: damp each 3x3 block using its OWN trace, so both blocks get
    # comparable proportional regularization.
    trans_trace = np.trace(JTJ[0:3, 0:3])
    rot_trace = np.trace(JTJ[3:6, 3:6])

    damping = np.zeros(6)
    damping[0:3] = 1e-4 * trans_trace / 3.0 + 1e-8
    damping[3:6] = 1e-4 * rot_trace / 3.0 + 1e-8

    JTJ_damped = JTJ + np.diag(damping)

    try:
        cov = sigma2 * np.linalg.inv(JTJ_damped)
    except np.linalg.LinAlgError:
        return None

    cov = 0.5 * (cov + cov.T)
    return cov


def estimate_point_to_point_covariance(points_i, points_j, T,
                                        voxel_size=0.5,
                                        correspondence_dist_threshold=1.0,
                                        min_correspondences=20,
                                        max_correspondences=300):
    """
    Point-to-point version: residual is the FULL 3D gap between each
    correspondence (not just the normal-direction component), so no
    surface normals are needed here. Uses the same downsampling +
    correspondence-capping + block-specific damping as the point-to-plane
    version, for a fair comparison between the two methods.
    """
    pc_i = o3d.geometry.PointCloud()
    pc_i.points = o3d.utility.Vector3dVector(points_i)
    pc_i = pc_i.voxel_down_sample(voxel_size)
    points_i_ds = np.asarray(pc_i.points)

    pc_j = o3d.geometry.PointCloud()
    pc_j.points = o3d.utility.Vector3dVector(points_j)
    pc_j = pc_j.voxel_down_sample(voxel_size)
    points_j_ds = np.asarray(pc_j.points)

    R = T[:3, :3]
    t = T[:3, 3]
    transformed = (R @ points_j_ds.T).T + t

    tree = cKDTree(points_i_ds)
    dists, idxs = tree.query(transformed, k=1)

    valid = np.where(dists < correspondence_dist_threshold)[0]
    if len(valid) < min_correspondences:
        return None

    if len(valid) > max_correspondences:
        valid = np.random.choice(valid, max_correspondences, replace=False)

    tgt_valid = points_i_ds[idxs[valid]]
    transformed_valid = transformed[valid]

    n = tgt_valid.shape[0]
    residuals = np.zeros(3 * n)
    J = np.zeros((3 * n, 6))

    for i in range(n):
        p_t = transformed_valid[i]
        q = tgt_valid[i]

        residuals[3 * i:3 * i + 3] = q - p_t
        J[3 * i:3 * i + 3, 0:3] = -np.eye(3)
        J[3 * i:3 * i + 3, 3:6] = skew(p_t)

    JTJ = J.T @ J
    sigma2 = np.sum(residuals ** 2) / max(3 * n - 6, 1)

    # Same block-specific damping fix as the point-to-plane version --
    # translation and rotation blocks can differ by orders of magnitude
    # here too, for the same reason (skew(p_t) scales with point range).
    trans_trace = np.trace(JTJ[0:3, 0:3])
    rot_trace = np.trace(JTJ[3:6, 3:6])

    damping = np.zeros(6)
    damping[0:3] = 1e-4 * trans_trace / 3.0 + 1e-8
    damping[3:6] = 1e-4 * rot_trace / 3.0 + 1e-8

    JTJ_damped = JTJ + np.diag(damping)

    try:
        cov = sigma2 * np.linalg.inv(JTJ_damped)
    except np.linalg.LinAlgError:
        return None

    cov = 0.5 * (cov + cov.T)
    return cov


def covariance_to_info_upper_triangle(cov_6x6):
    info = np.linalg.inv(cov_6x6)
    info = 0.5 * (info + info.T)
    upper = []
    for i in range(6):
        for j in range(i, 6):
            upper.append(info[i, j])
    return upper


def main():
    if len(sys.argv) < 4:
        print("Usage: python3 Covariance_estimation.py <g2o_file> <lidar_data_dir> <calib_imu_to_velo.txt>")
        print("  Writes two output files: <g2o_file>_p2plane.g2o and <g2o_file>_p2point.g2o")
        sys.exit()

    g2o_path = sys.argv[1]
    lidar_input_file_path = sys.argv[2]
    calib_path = sys.argv[3]

    base = g2o_path[:-4] if g2o_path.endswith(".g2o") else g2o_path
    output_path_p2plane = base + "_p2plane.g2o"
    output_path_p2point = base + "_p2point.g2o"

    print("Loading scans...")
    scans = load_all_scans(lidar_input_file_path)
    print(f"Loaded {len(scans)} scans")

    print("Loading IMU-to-Velodyne calibration...")
    C_imu_to_velo = load_imu_to_velo_calib(calib_path)

    print("Parsing g2o file...")
    vertices, edges = parse_g2o_file(g2o_path)
    print(f"Found {len(vertices)} vertices, {len(edges)} edges (before adding missing odometry edges)")

    edges = ensure_odometry_edges(vertices, edges)
    print(f"Total edges to process: {len(edges)}")

    vertex_lines = []
    for vid in sorted(vertices.keys()):
        x, y, z, qw, qx, qy, qz = vertices[vid]
        vertex_lines.append(
            f"VERTEX_SE3:QUAT {vid} {x} {y} {z} {qx} {qy} {qz} {qw}")

    output_lines_p2plane = list(vertex_lines)
    output_lines_p2point = list(vertex_lines)

    skipped = 0
    min_icp_fitness = 0.3

    for edge in edges:
        id1, id2 = edge["id1"], edge["id2"]

        if id1 >= len(scans) or id2 >= len(scans):
            print(f"  Skipped edge ({id1}, {id2}): index out of range")
            skipped += 1
            continue

        points_i = scans[id1]
        points_j = scans[id2]

        R_init = quat_to_rotation_matrix(edge["qx"], edge["qy"], edge["qz"], edge["qw"])
        T_init = np.eye(4)
        T_init[:3, :3] = R_init
        T_init[:3, 3] = [edge["dx"], edge["dy"], edge["dz"]]
        T_init = transform_delta_to_velo_frame(T_init, C_imu_to_velo)

        T_refined, fitness = run_icp_refine(points_i, points_j, T_init)

        if fitness < min_icp_fitness:
            print(f"  Skipped edge ({id1}, {id2}): ICP fitness {fitness:.3f} too low")
            skipped += 1
            continue

        # Same ICP-refined pose used for both -- only the covariance
        # method differs, so any downstream difference in results comes
        # purely from the covariance estimation method itself.
        cov_p2plane = estimate_point_to_plane_covariance(points_i, points_j, T_refined)
        cov_p2point = estimate_point_to_point_covariance(points_i, points_j, T_refined)

        if cov_p2plane is None or cov_p2point is None:
            print(f"  Skipped edge ({id1}, {id2}): covariance estimation failed")
            skipped += 1
            continue

        R_refined = T_refined[:3, :3]
        t_refined = T_refined[:3, 3]
        qx, qy, qz, qw = rotation_matrix_to_quat(R_refined)

        pose_str = (f"{t_refined[0]:.6f} {t_refined[1]:.6f} {t_refined[2]:.6f} "
                    f"{qx:.6f} {qy:.6f} {qz:.6f} {qw:.6f}")

        info_plane_str = " ".join(f"{v:.6f}" for v in covariance_to_info_upper_triangle(cov_p2plane))
        info_point_str = " ".join(f"{v:.6f}" for v in covariance_to_info_upper_triangle(cov_p2point))

        output_lines_p2plane.append(f"EDGE_SE3:QUAT {id1} {id2} {pose_str} {info_plane_str}")
        output_lines_p2point.append(f"EDGE_SE3:QUAT {id1} {id2} {pose_str} {info_point_str}")

        print(f"  Edge ({id1}, {id2}): ICP fitness={fitness:.3f}, both covariances computed")

    with open(output_path_p2plane, "w") as f:
        for line in output_lines_p2plane:
            f.write(line + "\n")

    with open(output_path_p2point, "w") as f:
        for line in output_lines_p2point:
            f.write(line + "\n")

    print(f"\nDone. {len(edges) - skipped} edges written, {skipped} skipped.")
    print(f"Wrote point-to-plane version to {output_path_p2plane}")
    print(f"Wrote point-to-point version to {output_path_p2point}")


if __name__ == "__main__":
    main()