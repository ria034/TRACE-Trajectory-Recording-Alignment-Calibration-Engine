import sys
import os
import numpy as np
import open3d as o3d


HIT_DTYPE_SIZE = 8  # 3x uint16 (x,y,z) + 2x uint8 (intensity, ring)
SCALE = 0.005
OFFSET = -100.0


def load_synced_scan(filepath):
    import struct
    with open(filepath, "rb") as f:
        data = f.read()
    n_hits = len(data) // HIT_DTYPE_SIZE
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
    """Lists scans directly from velodyne_sync, sorted by filename (=utime)."""

    def __init__(self, sync_dir):
        self.sync_dir = sync_dir
        self.filenames = sorted(f for f in os.listdir(sync_dir) if f.endswith(".bin"))

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        filepath = os.path.join(self.sync_dir, self.filenames[index])
        return load_synced_scan(filepath)


def load_candidates(candidates_path):
    """
    Reads a candidates file in the format saved by
    Loop_closure_detection_nclt.py's checkpointing:
        nn_idx,idx
        nn_idx,idx
        ...
    Returns a list of (nn_idx, idx) integer tuples.
    """
    candidates = []
    with open(candidates_path, "r") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) != 2:
                continue
            candidates.append((int(parts[0]), int(parts[1])))
    return candidates


def run_icp(points_i, points_j, T_init=None, voxel_size=0.5, max_correspondence_distance=1.0):
    if T_init is None:
        T_init = np.eye(4)

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


def apply_loop_closures(candidates, scan_loader, odometric_file_path,
                         min_icp_fitness=0.3, checkpoint_every=200):
    """
    Runs ICP for each (nn_idx, idx) candidate pair and appends the
    resulting EDGE_SE3:QUAT lines (with a placeholder information matrix
    -- real covariance gets computed later by Covariance_estimation_nclt.py)
    to the g2o file. Checkpoints progress periodically so a crash never
    loses all the ICP work again.
    """
    written = 0
    skipped = 0

    with open(odometric_file_path, "a") as f:
        for count, (scan_i, scan_j) in enumerate(candidates):
            points_i = scan_loader[scan_i]
            points_j = scan_loader[scan_j]

            T, fitness = run_icp(points_i, points_j)

            if fitness < min_icp_fitness:
                print(f"  Skipped ({scan_i}, {scan_j}): ICP fitness {fitness:.3f} too low")
                skipped += 1
                continue

            R = T[:3, :3]
            t = T[:3, 3]
            dx, dy, dz = t[0], t[1], t[2]
            qx, qy, qz, qw = rotation_matrix_to_quaternion(R)

            info_values = [500.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                            500.0, 0.0, 0.0, 0.0, 0.0,
                            500.0, 0.0, 0.0, 0.0,
                            500.0, 0.0, 0.0,
                            500.0, 0.0,
                            500.0]
            info_str = " ".join(str(v) for v in info_values)

            f.write(f"EDGE_SE3:QUAT {scan_i} {scan_j} "
                    f"{dx} {dy} {dz} {qx} {qy} {qz} {qw} {info_str}\n")
            f.flush()
            written += 1

            print(f"  Wrote edge ({scan_i}, {scan_j}), fitness={fitness:.3f} "
                  f"[{count+1}/{len(candidates)}]")

    print(f"\nDone. {written} edges written, {skipped} skipped "
          f"(ICP fitness too low), out of {len(candidates)} candidates.")


def main():
    if len(sys.argv) < 4:
        print("Usage: python3 apply_loop_closures_from_candidates.py "
              "<candidates_file> <velodyne_sync_dir> <odometric_g2o_file>")
        print("  candidates_file: lines of 'nn_idx,idx' (same format as the")
        print("  .loop_candidates.checkpoint file saved by Loop_closure_detection_nclt.py)")
        sys.exit()

    candidates_path = sys.argv[1]
    sync_dir = sys.argv[2]
    odometric_file_path = sys.argv[3]

    print(f"Loading candidates from {candidates_path}...")
    candidates = load_candidates(candidates_path)
    print(f"Loaded {len(candidates)} candidates")

    print(f"Setting up scan loader...")
    scan_loader = DirectScanLoader(sync_dir)
    print(f"Scan loader ready: {len(scan_loader)} scans")

    apply_loop_closures(candidates, scan_loader, odometric_file_path)


if __name__ == "__main__":
    main()