import sys
import os
import glob
from math import pi, cos, log, tan, sin

earth_radius = 6378137.0

lat0 = None
scale = None
x0, y0, z0 = None, None, None  # Local origin reference anchor


def read_file_and_convert_data(filepath):
    global lat0, scale, x0, y0, z0

    with open(filepath, "r") as f:
        line = f.readline()
        values = line.strip().split()
        values = [float(v) for v in values]

    latitude = values[0]
    longitude = values[1]
    altitude = values[2]
    roll = values[3]
    pitch = values[4]
    yaw = values[5]

    if lat0 is None:
        lat0 = latitude
        scale = cos(lat0 * pi / 180.0)

    # 1. Compute Mercator coordinates
    x_abs = (scale * longitude * pi * earth_radius) / 180.0
    y_abs = scale * earth_radius * log(tan((90.0 + latitude) * pi / 360.0))
    z_abs = altitude

    # 2. Anchor VERTEX 0 as local origin (0, 0, 0)
    if x0 is None:
        x0, y0, z0 = x_abs, y_abs, z_abs

    # 3. Express trajectory poses in local world frame
    x = x_abs - x0
    y = y_abs - y0
    z = z_abs - z0

    return x, y, z, roll, pitch, yaw


def convert_angles_to_quat(roll, pitch, yaw):
    cr = cos(roll / 2.0)
    sr = sin(roll / 2.0)
    cp = cos(pitch / 2.0)
    sp = sin(pitch / 2.0)
    cy = cos(yaw / 2.0)
    sy = sin(yaw / 2.0)

    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy

    return qw, qx, qy, qz


def write_vertex_to_g2o_file(output_file, vertex_id, x, y, z, qw, qx, qy, qz):
    with open(output_file, "a") as f:
        f.write(f"VERTEX_SE3:QUAT {vertex_id} {x:.6f} {y:.6f} {z:.6f} {qx:.6f} {qy:.6f} {qz:.6f} {qw:.6f}\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 oxts_to_g2o_fixed.py <path_to_oxts_data_folder> [output_directory]")
        sys.exit(1)

    oxts_dir = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "."

    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "kitti_0009.g2o")

    # Clear/initialize the output file
    open(output_file, "w").close()

    filepaths = sorted(glob.glob(f"{oxts_dir}/*.txt"))

    for vertex_id, filepath in enumerate(filepaths):
        x, y, z, roll, pitch, yaw = read_file_and_convert_data(filepath)
        qw, qx, qy, qz = convert_angles_to_quat(roll, pitch, yaw)
        write_vertex_to_g2o_file(output_file, vertex_id, x, y, z, qw, qx, qy, qz)

    print(f"Wrote {len(filepaths)} centered vertices to {output_file}") 