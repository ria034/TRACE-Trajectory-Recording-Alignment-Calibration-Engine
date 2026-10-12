import sys
import struct
import os
import numpy as np


# --- UNVERIFIED FORMAT ASSUMPTION -- confirm against NCLT's own devkit if
# you can find it (commonly distributed as read_vel_hits.py or similar).
# Documented format (best recollection, not independently verified here):
#   Each hit = 8 bytes:
#     x: uint16 (2 bytes)
#     y: uint16 (2 bytes)
#     z: uint16 (2 bytes)
#     intensity: uint8 (1 byte)
#     ring/laser_id: uint8 (1 byte)
#   Scaling: actual_meters = raw_uint16 * 0.005 - 100.0
# If the sanity-check ranges printed below look wrong (e.g. everything
# clustered near one value, or wildly outside +/-100m), this assumption
# is likely incorrect and needs to be revised against a verified source.
HIT_STRUCT = struct.Struct("<HHHBB")  # little-endian: 3x uint16, 2x uint8
HIT_SIZE = HIT_STRUCT.size  # should be 8
SCALE = 0.005
OFFSET = -100.0


def load_synced_scan(filepath):
    """
    Reads one file from NCLT's velodyne_sync folder -- this is already a
    single, complete scan (no manual binning needed, unlike the raw
    velodyne_hits.bin stream). Returns an Nx5 array: [x, y, z, intensity, ring].
    """
    with open(filepath, "rb") as f:
        data = f.read()

    n_hits = len(data) // HIT_SIZE
    if n_hits * HIT_SIZE != len(data):
        print(f"WARNING: {filepath} size ({len(data)} bytes) is not an exact "
              f"multiple of {HIT_SIZE} -- format assumption may be wrong, or "
              f"file is truncated/corrupted.")

    hits = np.zeros((n_hits, 5))
    for i in range(n_hits):
        chunk = data[i * HIT_SIZE:(i + 1) * HIT_SIZE]
        x_raw, y_raw, z_raw, intensity, ring = HIT_STRUCT.unpack(chunk)
        hits[i, 0] = x_raw * SCALE + OFFSET
        hits[i, 1] = y_raw * SCALE + OFFSET
        hits[i, 2] = z_raw * SCALE + OFFSET
        hits[i, 3] = intensity
        hits[i, 4] = ring

    return hits


def load_all_synced_scans(sync_dir, limit=None):
    """
    Loads scan files in a velodyne_sync folder, sorted by filename
    (which sorts correctly by timestamp since NCLT's filenames ARE the
    utime, same digit-count convention as groundtruth).

    limit: if set, only loads the first `limit` files -- important for
    NCLT, since a full session can have tens of thousands of scans and
    loading them all into memory at once can exhaust RAM (unlike KITTI's
    much shorter ~270-scan sequences).

    Returns a list of (utime, Nx5_hits_array) tuples, in chronological order.
    """
    import os
    filenames = sorted(os.listdir(sync_dir))
    filenames = [f for f in filenames if f.endswith(".bin")]
    if limit is not None:
        filenames = filenames[:limit]

    scans = []
    for filename in filenames:
        utime = int(filename.replace(".bin", ""))
        filepath = os.path.join(sync_dir, filename)
        hits = load_synced_scan(filepath)
        scans.append((utime, hits))
    return scans


class NCLTScanLoader:
    """
    Lazy, on-demand loader for NCLT's velodyne_sync scans. Unlike KITTI's
    load_all_scans (which preloads every scan into one big Python list --
    fine for ~270 scans, but would need tens of GB of RAM for a full NCLT
    session with 20,000-30,000+ scans), this loads one scan from disk at
    a time, only when actually requested.

    Requires a scan_order.txt file (produced by
    sync_nclt_scans_to_groundtruth.py) listing scan filenames in the same
    order as the g2o vertex ids, so loader[5] corresponds to VERTEX_SE3:QUAT 5.
    """

    def __init__(self, sync_dir, scan_order_path):
        self.sync_dir = sync_dir
        with open(scan_order_path, "r") as f:
            self.filenames = [line.strip() for line in f if line.strip()]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        filepath = os.path.join(self.sync_dir, self.filenames[index])
        return load_synced_scan(filepath)


def load_velodyne_hits(filepath, max_hits=None):
    """
    Reads the raw hit stream and returns an Nx5 array:
        [x_meters, y_meters, z_meters, intensity, ring]
    with x/y/z already converted from raw scaled integers to meters.

    max_hits: optional cap, useful for a quick sanity check on a huge file
              before committing to reading the whole thing.
    """
    hits = []
    with open(filepath, "rb") as f:
        while True:
            chunk = f.read(HIT_SIZE)
            if len(chunk) < HIT_SIZE:
                break  # end of file (or trailing partial record)

            x_raw, y_raw, z_raw, intensity, ring = HIT_STRUCT.unpack(chunk)

            x = x_raw * SCALE + OFFSET
            y = y_raw * SCALE + OFFSET
            z = z_raw * SCALE + OFFSET

            hits.append((x, y, z, intensity, ring))

            if max_hits is not None and len(hits) >= max_hits:
                break

    return np.array(hits)


def sanity_check(hits):
    if len(hits) == 0:
        print("No hits parsed -- something is wrong with the file path or format.")
        return

    x, y, z, intensity, ring = hits[:, 0], hits[:, 1], hits[:, 2], hits[:, 3], hits[:, 4]

    print(f"Total hits parsed: {len(hits)}")
    print(f"\nx range: [{x.min():.3f}, {x.max():.3f}] meters")
    print(f"y range: [{y.min():.3f}, {y.max():.3f}] meters")
    print(f"z range: [{z.min():.3f}, {z.max():.3f}] meters")
    print(f"intensity range: [{intensity.min():.0f}, {intensity.max():.0f}]")
    print(f"ring/laser_id range: [{ring.min():.0f}, {ring.max():.0f}] "
          f"(unique values: {len(np.unique(ring))})")

    print("\n--- What to check ---")
    print("x/y/z should look like a plausible LiDAR range: roughly a few")
    print("meters to ~100m, both positive and negative (sensor at origin).")
    print("If everything is near a single value, or absurdly large/small,")
    print("the byte layout or scale/offset constants are likely wrong.")
    print("\nring/laser_id unique count should roughly match the sensor's")
    print("actual number of laser channels (NCLT used a 32-beam Velodyne")
    print("HDL-32E, so expect somewhere around 32 unique values, not")
    print("hundreds or just 1-2).")


def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  Raw hits stream: python3 nclt_velodyne_parser.py <velodyne_hits.bin> [max_hits]")
        print("  Synced folder:   python3 nclt_velodyne_parser.py --sync <velodyne_sync_dir>")
        sys.exit()

    if sys.argv[1] == "--sync":
        sync_dir = sys.argv[2]
        limit = int(sys.argv[3]) if len(sys.argv) > 3 else 10
        print(f"Loading first {limit} scans from {sync_dir} (for a quick sanity "
              f"check -- pass a 3rd argument to load more, or a full pipeline "
              f"run later would need a streaming approach instead of loading "
              f"everything into memory at once)...")
        scans = load_all_synced_scans(sync_dir, limit=limit)
        print(f"Loaded {len(scans)} scans")

        if len(scans) > 0:
            print(f"\nFirst scan: utime={scans[0][0]}, {scans[0][1].shape[0]} points")
            print(f"Last scan:  utime={scans[-1][0]}, {scans[-1][1].shape[0]} points")
            print("\nSanity check on first scan:")
            sanity_check(scans[0][1])
        return

    filepath = sys.argv[1]
    max_hits = int(sys.argv[2]) if len(sys.argv) > 2 else 100000

    print(f"Reading up to {max_hits} hits for a quick sanity check "
          f"(pass a larger number, or omit for a rough default, to read more)...")
    hits = load_velodyne_hits(filepath, max_hits=max_hits)
    sanity_check(hits)


if __name__ == "__main__":
    main()