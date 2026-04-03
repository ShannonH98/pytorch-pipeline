import os
import nibabel as nib
import dicom2nifti

dicom_dir = "/Users/shannonhenry/Desktop/pytorch-pipeline/nii"
output_dir = "/Users/shannonhenry/Desktop/pytorch-pipeline/nii_converted"

os.makedirs(output_dir, exist_ok=True)

# Convert the DICOM folder to NIfTI
print(f"Converting DICOMs in {dicom_dir} -> {output_dir} ...")
dicom2nifti.convert_directory(dicom_dir, output_dir, compression=True, reorient=True)

# List converted files
nii_files = [f for f in os.listdir(output_dir) if f.endswith(".nii") or f.endswith(".nii.gz")]
print(f"\nConverted {len(nii_files)} NIfTI file(s):")
for f in nii_files:
    print(f" - {f}")

# Load and inspect each one
for fname in nii_files:
    path = os.path.join(output_dir, fname)
    img = nib.load(path)
    data = img.get_fdata()
    print(f"\n{fname}")
    print(f"  Shape:  {data.shape}")
    print(f"  Voxel size: {img.header.get_zooms()}")
    print(f"  dtype:  {data.dtype}")
    print(f"  min/max: {data.min():.2f} / {data.max():.2f}")
