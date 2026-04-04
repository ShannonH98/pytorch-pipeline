import torch
from sklearn.metrics import confusion_matrix, classification_report


def calculate_accuracy(outputs, labels):
    _, preds = torch.max(outputs, 1)
    correct = (preds == labels).sum().item()
    return correct / len(labels)


def evaluate(model, loader, criterion, device):
    """
    Runs model over a dataloader, returns loss, accuracy, and all predictions.
    """
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / len(loader)
    accuracy = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)

    return avg_loss, accuracy, all_preds, all_labels


def print_evaluation(all_preds, all_labels, class_names):
    print("\nConfusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))
    print("\nPer-class Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))
