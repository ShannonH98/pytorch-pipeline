import nibabel as nib
import numpy as np
from PIL import Image
import os

def nii_to_slices(nii_path, output_folder):
    # 1. Load file
    img = nib.load(nii_path)
    volume = img.get_fdata()

    print("Shape:", volume.shape)


    os.makedirs(output_folder, exist_ok=True)


    for i in range(volume.shape[2]):
        slice_i = volume[:, :, i]

        # Skip empty slices
        if np.max(slice_i) == 0:
            continue

        # Normalize
        slice_img = (slice_i / np.max(slice_i) * 255).astype(np.uint8)

        # Save image
        Image.fromarray(slice_img).save(
            os.path.join(output_folder, f"slice_{i:03d}.png")
        )

    print("Done!")