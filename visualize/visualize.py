import matplotlib.pyplot as plt
import nibabel as nib
from PIL import Image

img = Image.open("../data/slices/slice_010.png")
plt.imshow(img, cmap="gray")
plt.axis("off")
plt.show()

path = "/Users/shannonhenry/Desktop/pytorch-pipeline/nii_converted/201_t2w_tse.nii.gz"

img_nii = nib.load(path)
print("Loaded:", img_nii)

volume = img_nii.get_fdata()

fig, axis = plt.subplots(3, 3, figsize=(10, 10))

slice_counter = 0

for i in range(3):
    for j in range(3):
        axis[i, j].imshow(volume[:, :, slice_counter], cmap="gray")
        axis[i, j].axis("off")
        slice_counter += 1

plt.show() 