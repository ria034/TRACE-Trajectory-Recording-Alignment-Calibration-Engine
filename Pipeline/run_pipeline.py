import sys
import os
import time
import subprocess
import argparse

def run_cmd(cmd):
    """Utility to run shell/python commands with real-time streaming logs."""
    print(f"\n==================================================")
    print(f" Executing: {' '.join(cmd)}")
    print(f"==================================================\n")
    start_time = time.time()
    
    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
    )
    
    for line in iter(process.stdout.readline, ''):
        print(line, end='')
        
    process.stdout.close()
    return_code = process.wait()
    
    elapsed = time.time() - start_time
    if return_code != 0:
        print(f"\n[ERROR] Command failed with exit code {return_code}")
        sys.exit(return_code)
        
    print(f"\n[SUCCESS] Completed in {elapsed:.2f} seconds.")

def main():
    parser = argparse.ArgumentParser(
        description="Master Pipeline: Raw OXTS/Velodyne Data -> g2o File with ICP Loop Closures and Point-to-Plane Covariances."
    )
    # Support both --oxts-dir and --oxts_dir flag formats with raw string defaults
    parser.add_argument(
        "--oxts-dir", "--oxts_dir", 
        default=r"C:\Users\lenovo\Downloads\Pose_graph_sparsification_research\g2o_file_generation\Data\kitti_0018\2011_09_26\2011_09_26_drive_0018_sync\oxts\data", 
        help="Path to raw OXTS text/csv directory"
    )
    parser.add_argument(
        "--lidar-dir", "--lidar_dir", 
        default=r"C:\Users\lenovo\Downloads\Pose_graph_sparsification_research\g2o_file_generation\Data\kitti_0018\2011_09_26\2011_09_26_drive_0018_sync\velodyne_points\data", 
        help="Path to raw Velodyne .bin scans directory"
    )
    parser.add_argument(
        "--output-g2o", "--output_g2o", 
        default=r"C:\Users\lenovo\Downloads\Pose_graph_sparsification_research\g2o_file_generation\Output\kitti_0018.g2o", 
        help="Path for the generated output .g2o file"
    )
    
    # Defaults set to your exact full script paths on disk using raw strings
    parser.add_argument(
        "--script1", 
        default=r"C:\Users\lenovo\Downloads\Pose_graph_sparsification_research\g2o_file_generation\Pipeline\Odometric_processing.py", 
        help="Stage 1 script path"
    )
    parser.add_argument(
        "--script2", 
        default=r"C:\Users\lenovo\Downloads\Pose_graph_sparsification_research\g2o_file_generation\Pipeline\Loop_closure_detection.py", 
        help="Stage 2 script path"
    )
    parser.add_argument(
        "--script3", 
        default=r"C:\Users\lenovo\Downloads\Pose_graph_sparsification_research\g2o_file_generation\Pipeline\Covariance_estimation.py", 
        help="Stage 3 script path"
    )

    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))

    oxts_dir = os.path.abspath(args.oxts_dir)
    lidar_dir = os.path.abspath(args.lidar_dir)
    output_g2o = os.path.abspath(args.output_g2o)

    # Ensure output target is a file, not a directory
    if os.path.isdir(output_g2o):
        output_g2o = os.path.join(output_g2o, "kitti_0018.g2o")
        print(f"[INFO] Output path was a directory. Targeting file: {output_g2o}")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_g2o), exist_ok=True)

    # Resolve stage script paths
    stage1_script = os.path.abspath(args.script1) if os.path.isabs(args.script1) else os.path.join(script_dir, args.script1)
    stage2_script = os.path.abspath(args.script2) if os.path.isabs(args.script2) else os.path.join(script_dir, args.script2)
    stage3_script = os.path.abspath(args.script3) if os.path.isabs(args.script3) else os.path.join(script_dir, args.script3)

    for name, script_path in [("Stage 1", stage1_script), ("Stage 2", stage2_script), ("Stage 3", stage3_script)]:
        if not os.path.exists(script_path):
            raise FileNotFoundError(f"[ERROR] {name} script not found at: '{script_path}'")

    pipeline_start = time.time()

    # --- STEP 1: Parse OXTS ---
    print("\n>>> STAGE 1/3: Converting raw OXTS readings to base g2o trajectory...")
    cmd_stage1 = [sys.executable, stage1_script, oxts_dir, output_g2o]
    run_cmd(cmd_stage1)

    # --- STEP 2: Detect Loop Closures ---
    print("\n>>> STAGE 2/3: Detecting loop closures with ScanContext and refining via ICP...")
    cmd_stage2 = [sys.executable, stage2_script, output_g2o, lidar_dir]
    run_cmd(cmd_stage2)

    # --- STEP 3: Compute Covariances ---
    print("\n>>> STAGE 3/3: Evaluating Point-to-Plane Jacobians and computing full 6-DoF Covariances...")
    cmd_stage3 = [sys.executable, stage3_script, output_g2o, lidar_dir]
    run_cmd(cmd_stage3)

    total_elapsed = time.time() - pipeline_start
    print(f"\n==================================================")
    print(f" PIPELINE COMPLETE ")
    print(f" Output G2O File : {output_g2o}")
    print(f" Total Runtime   : {total_elapsed / 60.0:.2f} minutes")
    print(f"==================================================\n")

if __name__ == "__main__":
    main()