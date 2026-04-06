from torchvision import datasets
from torch.utils.data import DataLoader, random_split
from preprocessing.preprocess import get_transforms
import os


def get_dataloaders(data_dir="data", batch_size=32, val_split=0.15):
    train_transform, test_transform = get_transforms()

    # Test loader
    test_dataset = datasets.ImageFolder(root=f"{data_dir}/test", transform=test_transform)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Val loader — use separate folder if it exists, otherwise split from train
    if os.path.isdir(f"{data_dir}/val"):
        train_dataset = datasets.ImageFolder(root=f"{data_dir}/train", transform=train_transform)
        val_dataset   = datasets.ImageFolder(root=f"{data_dir}/val",   transform=test_transform)
    else:
        full_train = datasets.ImageFolder(root=f"{data_dir}/train", transform=train_transform)
        n_val   = int(len(full_train) * val_split)
        n_train = len(full_train) - n_val
        train_dataset, val_dataset = random_split(full_train, [n_train, n_val])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
