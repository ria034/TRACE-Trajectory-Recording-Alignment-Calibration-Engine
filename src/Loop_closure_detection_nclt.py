import sys
sys.path.insert(0, r"C:\Users\lenovo\Downloads\Pose_graph_sparsification_research\g2o_file_generation\Pipeline\PyICP-SLAM")
sys.path.insert(0, r"C:\Users\lenovo\Downloads\Pose_graph_sparsification_research\g2o_file_generation\Pipeline\PyICP-SLAM\utils")
import os
import numpy as np
from ScanContextManager import ScanContextManager
import open3d as o3d

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


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


def save_candidates_checkpoint(loop_candidates, checkpoint_path):
    with open(checkpoint_path, "w") as f:
        for nn_idx, idx in loop_candidates:
            f.write(f"{nn_idx},{idx}\n")


def load_candidates_checkpoint(checkpoint_path):
    candidates = []
    with open(checkpoint_path, "r") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) != 2:
                continue
            candidates.append((int(parts[0]), int(parts[1])))
    return candidates


def detect_loop_closures(scan_loader, checkpoint_path=None, checkpoint_every=1000):
    """
    Same Scan Context logic as the KITTI version, but pulls each scan
    lazily from the loader (one at a time) instead of iterating over a
    fully preloaded list -- essential for NCLT's much larger scan counts.

    checkpoint_path: if given, saves found candidates to disk every
    checkpoint_every scans, so a crash never loses all progress up to
    that point (e.g. ScanContextManager hitting an internal capacity
    limit on very large sequences).
    """
    scm = ScanContextManager(shape=[20, 60], num_candidates=10, threshold=0.15)
    loop_candidates = []

    for idx in range(len(scan_loader)):
        scan = scan_loader[idx][:, :3]  # drop intensity/ring, keep x,y,z
        scm.addNode(idx, scan)
        nn_idx, nn_dist, nn_yawdiff_deg = scm.detectLoop()
        if nn_idx is not None:
            loop_candidates.append((nn_idx, idx))

        if (idx + 1) % checkpoint_every == 0:
            print(f"  Processed {idx + 1} / {len(scan_loader)} scans, "
                  f"{len(loop_candidates)} loop closure candidates so far")
            if checkpoint_path is not None:
                try:
                    save_candidates_checkpoint(loop_candidates, checkpoint_path)
                    print(f"    (checkpoint saved to {checkpoint_path})")
                except Exception as e:
                    print(f"    WARNING: checkpoint save failed: {e}")

    return loop_candidates


def run_icp(points_i, points_j):
    source = o3d.geometry.PointCloud()
    source.points = o3d.utility.Vector3dVector(points_j)
    target = o3d.geometry.PointCloud()
    target.points = o3d.utility.Vector3dVector(points_i)

    source_down = source.voxel_down_sample(voxel_size=0.5)
    target_down = target.voxel_down_sample(voxel_size=0.5)
    target_down.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1.0, max_nn=30))

    result = o3d.pipelines.registration.registration_icp(
        source_down, target_down,
        max_correspondence_distance=0.5,
        init=np.eye(4),
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50)
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


def write_loop_closures(loop_candidates, scan_loader, odometric_file_path):
    with open(odometric_file_path, "a") as f:
        for scan_i, scan_j in loop_candidates:
            points_i = scan_loader[scan_i][:, :3]
            points_j = scan_loader[scan_j][:, :3]

            T = run_icp(points_i, points_j)
            if T is None:
                print(f"  Skipped ({scan_i}, {scan_j}): ICP fit too poor")
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

            print(f"  Wrote edge ({scan_i}, {scan_j})")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 Loop_closure_detection_nclt.py "
              "<odometric_g2o_file> <velodyne_sync_dir>")
        sys.exit()

    odometric_file_path = sys.argv[1]
    sync_dir = sys.argv[2]

    scan_loader = DirectScanLoader(sync_dir)
    print(f"Scan loader ready: {len(scan_loader)} scans (loaded on demand, not all at once)")

    candidates_checkpoint_path = odometric_file_path + ".loop_candidates.checkpoint"

    if os.path.exists(candidates_checkpoint_path):
        print(f"\nFound existing candidate checkpoint: {candidates_checkpoint_path}")
        print("NOTE: this resumes the CANDIDATE LIST, but Scan Context detection")
        print("itself will still re-run from scan 0 (ScanContextManager does not")
        print("support resuming its own internal descriptor state). If the crash")
        print("was due to a capacity limit, that needs to be fixed first, or this")
        print("will hit the same crash again at the same scan index.")
        loop_candidates = load_candidates_checkpoint(candidates_checkpoint_path)
        print(f"Loaded {len(loop_candidates)} previously-found candidates "
              f"(will be combined with any newly found ones)")
    else:
        loop_candidates = []

    new_candidates = detect_loop_closures(
        scan_loader, checkpoint_path=candidates_checkpoint_path)

    # Merge, avoiding duplicates if this run re-covers scans already checkpointed
    all_candidates = list(set(loop_candidates) | set(new_candidates))
    print(f"Found {len(all_candidates)} total loop closure candidates "
          f"(after merging with any checkpoint)")

    write_loop_closures(all_candidates, scan_loader, odometric_file_path)
    print(f"Done. Loop closure edges appended to {odometric_file_path}")