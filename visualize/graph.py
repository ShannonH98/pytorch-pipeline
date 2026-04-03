import matplotlib.pyplot as plt
fig, axis = plt.subplots(3,3, figsize=(10,10))

slice_counter = 0
for i in range(3):
  for j in range(3):
    axis[i,j].imshow(nifti_files.get_fdata()[:,:,slice_counter], cmap="gray")
    slice_counter += 1