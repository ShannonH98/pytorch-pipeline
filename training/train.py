import torch
import torch.nn as nn
import torch.optim as optim

from models.cnn import BrainCNN
from utils.dataloader import get_dataloaders
from evaluation.metrics import calculate_accuracy, evaluate, print_evaluation

EPOCHS = 20
LR = 0.001
CHECKPOINT = "best_model.pth"


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load data
    train_loader, val_loader, test_loader = get_dataloaders()
    class_names = train_loader.dataset.classes
    print(f"Classes: {class_names}")
    print(f"Train: {len(train_loader.dataset)} | Val: {len(val_loader.dataset)} | Test: {len(test_loader.dataset)}")

    # 2. Model
    model = BrainCNN().to(device)

    # 3. Loss + optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # 4. Training loop
    best_val_acc = 0.0

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        total_acc = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_acc += calculate_accuracy(outputs, labels)

        train_loss = total_loss / len(train_loader)
        train_acc = total_acc / len(train_loader)

        # 5. Validation
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)

        print(f"Epoch {epoch+1}/{EPOCHS} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

        # 6. Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), CHECKPOINT)
            print(f"  -> Saved best model (val acc: {val_acc:.4f})")

    # 7. Final test evaluation
    print("\n--- Test Evaluation ---")
    model.load_state_dict(torch.load(CHECKPOINT))
    test_loss, test_acc, preds, labels = evaluate(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")
    print_evaluation(preds, labels, class_names)


if __name__ == "__main__":
    train()
