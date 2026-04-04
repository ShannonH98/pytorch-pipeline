from torchvision import datasets
from torch.utils.data import DataLoader
from preprocessing.preprocess import get_transforms


def get_dataloaders(data_dir="data", batch_size=32):
    train_transform, test_transform = get_transforms()

    train_dataset = datasets.ImageFolder(root=f"{data_dir}/train", transform=train_transform)
    val_dataset   = datasets.ImageFolder(root=f"{data_dir}/val",   transform=test_transform)
    test_dataset  = datasets.ImageFolder(root=f"{data_dir}/test",  transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
