import sys
import numpy as np
import matplotlib.pyplot as plt


def load_vertices(g2o_path):
    """
    Parses both VERTEX_SE2 (2D) and VERTEX_SE3:QUAT (3D) lines.
    Returns (ids, positions, is_3d) sorted by vertex id.
    positions is Nx3 array [x, y, z] (z=0 for 2D).
    """
    verts = {}
    is_3d = False

    with open(g2o_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue

            # Handle 3D vertices
            if parts[0] in ("VERTEX_SE3:QUAT", "VERTEX_SE3"):
                is_3d = True
                vid = int(parts[1])
                x, y, z = float(parts[2]), float(parts[3]), float(parts[4])
                verts[vid] = (x, y, z)

            # Handle 2D vertices (VERTEX_SE2 x y theta)
            elif parts[0] == "VERTEX_SE2":
                vid = int(parts[1])
                x, y = float(parts[2]), float(parts[3])
                verts[vid] = (x, y, 0.0)

    ids = sorted(verts.keys())
    positions = np.array([verts[i] for i in ids]) if ids else np.empty((0, 3))
    return ids, positions, is_3d


def print_jump_stats(ids, positions, is_3d):
    diffs = np.diff(positions, axis=0)
    dists = np.linalg.norm(diffs if is_3d else diffs[:, :2], axis=1)

    print(f"Dimension mode: {'3D (SE3)' if is_3d else '2D (SE2)'}")
    print(f"Total vertices: {len(ids)}")
    print(f"Vertex id range: [{ids[0]}, {ids[-1]}]")
    print(f"\nConsecutive-vertex jump distances:")
    print(f"  mean={dists.mean():.3f} m  median={np.median(dists):.3f} m  "
          f"min={dists.min():.3f} m  max={dists.max():.3f} m")

    print(f"\nBounding box:")
    print(f"  x: [{positions[:,0].min():.2f}, {positions[:,0].max():.2f}]")
    print(f"  y: [{positions[:,1].min():.2f}, {positions[:,1].max():.2f}]")
    if is_3d:
        print(f"  z: [{positions[:,2].min():.2f}, {positions[:,2].max():.2f}]")

    total_path_length = dists.sum()
    straight_line = np.linalg.norm(positions[-1] - positions[0] if is_3d else (positions[-1] - positions[0])[:2])
    print(f"\nTotal path length: {total_path_length:.2f} m")
    print(f"Straight-line start-to-end distance: {straight_line:.2f} m")

    threshold = dists.mean() + 5 * dists.std()
    big_jumps = np.where(dists > threshold)[0]
    print(f"\nJumps larger than mean+5*std ({threshold:.2f} m): {len(big_jumps)}")
    if len(big_jumps) > 0:
        print("  First 10 large jumps (vertex id -> next vertex id, distance):")
        for idx in big_jumps[:10]:
            print(f"    {ids[idx]} -> {ids[idx+1]}: {dists[idx]:.2f} m")

    tiny = np.sum(dists < 0.01)
    print(f"\nJumps smaller than 1cm (possible duplicate matches): {tiny} / {len(dists)}")


def plot_trajectory(ids, positions, is_3d, output_path):
    if is_3d:
        fig = plt.figure(figsize=(15, 12))

        # 1. Top-Left: 2D Spatial Projection (XY)
        ax1 = fig.add_subplot(2, 2, 1)
        sc = ax1.scatter(positions[:, 0], positions[:, 1], c=range(len(ids)), cmap='viridis', s=10)
        ax1.plot(positions[:, 0], positions[:, 1], '-', linewidth=0.5, alpha=0.4, color='gray')
        ax1.plot(positions[0, 0], positions[0, 1], 'go', markersize=7, label='Start')
        ax1.plot(positions[-1, 0], positions[-1, 1], 'ro', markersize=7, label='End')
        ax1.set_xlabel("x (m)")
        ax1.set_ylabel("y (m)")
        ax1.set_title("2D Projection (X, Y)")
        ax1.set_aspect('equal', adjustable='datalim')
        ax1.legend()
        plt.colorbar(sc, ax=ax1, label="vertex index")

        # 2. Top-Right: 3D Trajectory Plot
        ax2 = fig.add_subplot(2, 2, 2, projection='3d')
        sc3d = ax2.scatter(positions[:, 0], positions[:, 1], positions[:, 2], 
                           c=range(len(ids)), cmap='viridis', s=10)
        ax2.plot(positions[:, 0], positions[:, 1], positions[:, 2], '-', linewidth=0.5, alpha=0.4, color='gray')
        ax2.plot([positions[0, 0]], [positions[0, 1]], [positions[0, 2]], 'go', markersize=8, label='Start')
        ax2.plot([positions[-1, 0]], [positions[-1, 1]], [positions[-1, 2]], 'ro', markersize=8, label='End')
        ax2.set_xlabel("x (m)")
        ax2.set_ylabel("y (m)")
        ax2.set_zlabel("z (m)")
        ax2.set_title("3D Trajectory View")
        ax2.legend()

        # 3. Bottom-Left: Z-Altitude Profile
        ax3 = fig.add_subplot(2, 2, 3)
        ax3.plot(range(len(ids)), positions[:, 2], '-', linewidth=1, color='tab:blue')
        ax3.set_xlabel("vertex index")
        ax3.set_ylabel("z (m)")
        ax3.set_title("Altitude Profile (Z over time)")
        ax3.grid(True, linestyle='--', alpha=0.5)

        # 4. Bottom-Right: 3D Step Distance Profile
        ax4 = fig.add_subplot(2, 2, 4)
        diffs = np.diff(positions, axis=0)
        step_dists = np.linalg.norm(diffs, axis=1)
        ax4.plot(range(len(ids) - 1), step_dists, '-', linewidth=0.8, color='tab:orange')
        ax4.set_xlabel("step index (i -> i+1)")
        ax4.set_ylabel("3D step distance (m)")
        ax4.set_title("Consecutive Step Sizes (3D)")
        ax4.grid(True, linestyle='--', alpha=0.5)

    else:
        # Fallback 1x2 layout for 2D Datasets
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        ax1 = axes[0]
        sc = ax1.scatter(positions[:, 0], positions[:, 1], c=range(len(ids)), cmap='viridis', s=15)
        ax1.plot(positions[:, 0], positions[:, 1], '-', linewidth=0.5, alpha=0.5, color='gray')
        ax1.plot(positions[0, 0], positions[0, 1], 'go', markersize=8, label='Start')
        ax1.plot(positions[-1, 0], positions[-1, 1], 'ro', markersize=8, label='End')
        ax1.set_xlabel("x (m)")
        ax1.set_ylabel("y (m)")
        ax1.set_title("2D Trajectory (X, Y)")
        ax1.set_aspect('equal', adjustable='datalim')
        ax1.legend()
        plt.colorbar(sc, ax=ax1, label="vertex index")

        ax2 = axes[1]
        diffs = np.diff(positions[:, :2], axis=0)
        step_dists = np.linalg.norm(diffs, axis=1)
        ax2.plot(range(len(ids) - 1), step_dists, '-', linewidth=1, color='tab:orange')
        ax2.set_xlabel("step index (i -> i+1)")
        ax2.set_ylabel("step distance (m)")
        ax2.set_title("Consecutive Step Sizes (2D)")
        ax2.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"\nSaved plot to {output_path}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 view_trajectory.py <g2o_file> [output_png]")
        sys.exit(1)

    g2o_path = sys.argv[1]
    output_png = sys.argv[2] if len(sys.argv) > 2 else "trajectory_plot.png"

    ids, positions, is_3d = load_vertices(g2o_path)
    if len(ids) == 0:
        print("No valid vertices found in file.")
        return

    print_jump_stats(ids, positions, is_3d)
    plot_trajectory(ids, positions, is_3d, output_png)
    plt.show()

if __name__ == "__main__":
    main()