import sys
import os


def truncate_g2o(input_path, output_path, max_vertex_id):
    with open(input_path, "r") as fin, open(output_path, "w") as fout:
        for line in fin:
            parts = line.strip().split()
            if not parts:
                continue
            if parts[0] == "VERTEX_SE3:QUAT":
                vid = int(parts[1])
                if vid <= max_vertex_id:
                    fout.write(line)
            elif parts[0] == "EDGE_SE3:QUAT":
                id1, id2 = int(parts[1]), int(parts[2])
                if id1 <= max_vertex_id and id2 <= max_vertex_id:
                    fout.write(line)


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python3 truncate_g2o.py <input_g2o> <output_dir> <max_vertex_id>")
        print("  output_dir is a FOLDER (created if it doesn't exist) -- the")
        print("  truncated file will be written as output_dir/truncated.g2o")
        sys.exit()

    input_path = sys.argv[1]
    output_dir = sys.argv[2]
    max_vertex_id = int(sys.argv[3])

    # The second argument is ALWAYS a folder now -- no more ambiguity
    # between "did I pass a file or a directory" (this exact mistake has
    # caused repeated PermissionErrors earlier in this project).
    if os.path.isfile(output_dir):
        print(f"ERROR: '{output_dir}' is an existing FILE, but this argument "
              f"must be a FOLDER (it will be created if it doesn't exist).")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "truncated.g2o")

    truncate_g2o(input_path, output_path, max_vertex_id)
    print(f"Truncated to vertex ids <= {max_vertex_id}, wrote {output_path}")