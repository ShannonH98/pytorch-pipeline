# pytorch-pipeline
Deep Learning for Neuroimaging (PyTorch Pipeline)

---

## Project Structure

```
pytorch-pipeline/
├── data/
│   ├── train/<label>/       ← training PNGs (one subfolder per class)
│   ├── val/<label>/         ← validation PNGs
│   └── test/<label>/        ← test PNGs
├── nii_converted/<label>/   ← NIfTI files organised by label (input to slicer)
├── models/
│   └── cnn.py               ← BrainCNN model
├── preprocessing/
│   ├── preprocess.py        ← image transforms
│   ├── nii_to_slices.py     ← NIfTI → PNG slice pipeline
│   └── nii_converter.py     ← DICOM → NIfTI converter
├── training/
│   └── train.py             ← training + validation + test loop
├── utils/
│   └── dataloader.py        ← train/val/test DataLoaders
├── evaluation/
│   └── metrics.py           ← accuracy, confusion matrix, per-class report
└── visualize/
    └── visualize.py
```

---

## How to run

### 1. Convert DICOMs to NIfTI
Drop DICOM series folders into `dicom2/` then run:
```bash
python preprocessing/nii_converter.py
```

### 2. Organise NIfTI files by label
Move converted `.nii.gz` files into labelled subfolders:
```
nii_converted/
  no_impairment/subject01.nii.gz
  mild_impairment/subject02.nii.gz
  moderate_impairment/subject03.nii.gz
  very_mild_impairment/subject04.nii.gz
```

### 3. Extract slices into train/val/test
```bash
python preprocessing/nii_to_slices.py
```
Splits by subject (70/15/15) and saves PNGs to `data/train`, `data/val`, `data/test`.

### 4. Train
```bash
python training/train.py
```
Trains for 20 epochs, validates each epoch, saves best model to `best_model.pth`, and prints a full test evaluation (confusion matrix + per-class accuracy) at the end.

---

## Files & Functions

| File | Function | What it does |
|------|----------|--------------|
| `models/cnn.py` | `BrainCNN` | 2-block conv net: Conv→ReLU→Pool ×2, then Linear(28800→4) for 4-class output |
| `preprocessing/preprocess.py` | `get_transforms()` | Train transforms (resize, flip, ToTensor) and test transforms (resize, ToTensor) |
| `preprocessing/nii_to_slices.py` | `run()` | Reads NIfTI files by label folder, splits subjects into train/val/test, saves 2D slices as PNGs |
| `preprocessing/nii_converter.py` | top-level script | Loops over DICOM subfolders in `dicom2/`, converts each series to `.nii.gz` in `nii_converted/` |
| `utils/dataloader.py` | `get_dataloaders()` | Returns train, val, and test `DataLoader`s using `ImageFolder` |
| `training/train.py` | `train()` | Full training loop: forward/backward pass, val each epoch, saves best checkpoint, final test eval |
| `evaluation/metrics.py` | `calculate_accuracy()` | Batch accuracy from model outputs vs labels |
| `evaluation/metrics.py` | `evaluate()` | Runs model over a full dataloader, returns loss, accuracy, and all predictions |
| `evaluation/metrics.py` | `print_evaluation()` | Prints confusion matrix and per-class classification report |

---

## Classes
- No Impairment
- Mild Impairment
- Moderate Impairment
- Very Mild Impairment
