import os
import numpy as np
from PIL import Image

output_folder = "data/mri/slices"
os.makedirs(output_folder, exist_ok=True)

for i in range(volume.shape[2]):  # loop through all slices
    slice_i = volume[:, :, i]
    
    # Normalize to 0-255
    slice_img = (slice_i / np.max(slice_i) * 255).astype(np.uint8)
    
    # Save as PNG
    slice_path = os.path.join(output_folder, f"slice_{i:03d}.png")
    Image.fromarray(slice_img).save(slice_path)

print(f"Saved {volume.shape[2]} slices to {output_folder}")