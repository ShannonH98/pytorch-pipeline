import subprocess
import os

def skull_strip(input_path, output_path):
    """
    Uses FSL's BET to remove non-brain tissue from an MRI scan.
    
    input_path: str -> path to raw MRI (.nii)
    output_path: str -> path to save brain-extracted MRI
    """
    try:
        subprocess.run(["bet", input_path, output_path], check=True)
        print(f"Skull stripping complete: {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error during skull stripping: {e}")

def preprocess_folder(input_folder, output_folder):
    """
    Apply skull stripping to all .nii files in a folder.
    """
    os.makedirs(output_folder, exist_ok=True)
    
    for filename in os.listdir(input_folder):
        if filename.endswith(".nii") or filename.endswith(".nii.gz"):
            in_path = os.path.join(input_folder, filename)
            out_path = os.path.join(output_folder, filename.replace(".nii", "_brain.nii"))
            skull_strip(in_path, out_path)