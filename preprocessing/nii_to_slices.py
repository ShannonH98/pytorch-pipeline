"""
NIfTI → PNG slice pipeline.

Expected input layout (one subfolder per label):
    nii_converted/
        no_impairment/
            subject01.nii.gz
            subject02.nii.gz
        mild_impairment/
            subject03.nii.gz
        ...

Output layout (split by subject, not by slice):
    data/
        train/<label>/slice_<subject>_<i>.png
        val/<label>/slice_<subject>_<i>.png
        test/<label>/slice_<subject>_<i>.png

Usage:
    python preprocessing/nii_to_slices.py
"""

import os
import random
import numpy as np
import nibabel as nib
from PIL import Image

NII_ROOT = "nii_converted"   # folder containing label subfolders
DATA_ROOT = "data"
SPLIT = (0.70, 0.15, 0.15)  # train / val / test
SEED = 42


def extract_slices(nii_path, subject_id, output_dir):
    img = nib.load(nii_path)
    volume = img.get_fdata()

    os.makedirs(output_dir, exist_ok=True)

    for i in range(volume.shape[2]):
        slc = volume[:, :, i]
        max_val = slc.max()
        if max_val == 0:
            continue  # skip empty slices
        slc_norm = (slc / max_val * 255).astype(np.uint8)
        fname = f"slice_{subject_id}_{i:03d}.png"
        Image.fromarray(slc_norm).save(os.path.join(output_dir, fname))


def run():
    random.seed(SEED)

    if not os.path.isdir(NII_ROOT):
        print(f"NII_ROOT '{NII_ROOT}' not found. Nothing to do.")
        return

    labels = [d for d in os.listdir(NII_ROOT) if os.path.isdir(os.path.join(NII_ROOT, d))]

    if not labels:
        print("No label subfolders found inside nii_converted/.")
        return

    for label in labels:
        label_dir = os.path.join(NII_ROOT, label)
        subjects = [
            f for f in os.listdir(label_dir)
            if f.endswith(".nii") or f.endswith(".nii.gz")
        ]

        if not subjects:
            print(f"  [{label}] no .nii.gz files found, skipping")
            continue

        random.shuffle(subjects)
        n = len(subjects)
        n_train = max(1, int(n * SPLIT[0]))
        n_val = max(1, int(n * SPLIT[1]))

        splits = {
            "train": subjects[:n_train],
            "val":   subjects[n_train:n_train + n_val],
            "test":  subjects[n_train + n_val:],
        }

        for split_name, split_subjects in splits.items():
            out_dir = os.path.join(DATA_ROOT, split_name, label)
            for fname in split_subjects:
                subject_id = fname.replace(".nii.gz", "").replace(".nii", "")
                nii_path = os.path.join(label_dir, fname)
                extract_slices(nii_path, subject_id, out_dir)
            print(f"  [{label}] {split_name}: {len(split_subjects)} subject(s)")

    print("\nDone. Slices saved to data/train, data/val, data/test.")


if __name__ == "__main__":
    run()
