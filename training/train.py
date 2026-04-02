import torch
import torch.nn as nn
import torch.optim as optim

from models.cnn import BrainCNN
from utils.dataloader import get_dataloaders
from evaluation.metrics import calculate_accuracy

def train():
    # 1. Load data
    train_loader, test_loader = get_dataloaders()

    # 2. Initialize model
    model = BrainCNN()

    # 3. Loss + optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 4. Training loop
    for epoch in range(5):
        total_loss = 0
        total_acc = 0

        model.train()

        for images, labels in train_loader:
            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            acc = calculate_accuracy(outputs, labels)

            total_loss += loss.item()
            total_acc += acc

        num_batches = len(train_loader)
        print(f"Epoch {epoch+1}")
        print(f"Loss: {total_loss / num_batches:.4f}, Accuracy: {total_acc / num_batches:.4f}")

    print("Training complete!")

if __name__ == "__main__":
    train()