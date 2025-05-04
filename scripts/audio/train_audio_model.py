import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from scripts.audio.utils.audio_utils import AudioEmotionModel
from scripts.audio.utils.data_loader import get_data_loaders
import os
from sklearn.metrics import classification_report
import numpy as np

# Configuration
CFG = {
    "batch_size": 32,
    "epochs": 34,
    "lr": 0.001,
    "input_size": 39,  
    "hidden_size": 256,
    "num_layers": 3,
    "num_classes": 7,
    "seq_length": 130
}

def train_model():
    os.makedirs("./models", exist_ok=True) 
    
    # Initialize data loaders
    train_loader, test_loader, _ = get_data_loaders("./datasets/audio_features.csv", CFG["batch_size"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Model setup
    model = AudioEmotionModel(
        CFG["input_size"], 
        CFG["hidden_size"],
        CFG["num_layers"],
        CFG["num_classes"]
    ).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=CFG["lr"])
    
    # Track metrics
    train_losses, test_losses = [], []
    train_accs, test_accs = [], []

    # Training loop
    for epoch in range(CFG["epochs"]):
        model.train()
        epoch_train_loss = 0.0
        correct_train = 0
        all_preds_train = []
        all_labels_train = []

        # Training phase
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(features)
            loss = criterion(outputs, labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Track metrics
            epoch_train_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            correct_train += (preds == labels).sum().item()
            all_preds_train.extend(preds.cpu().numpy())
            all_labels_train.extend(labels.cpu().numpy())

        # Calculate training metrics
        train_loss = epoch_train_loss / len(train_loader)
        train_acc = correct_train / len(train_loader.dataset)
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # Validation phase
        test_loss, test_acc, all_preds_test, all_labels_test = evaluate(model, test_loader, criterion, device)
        test_losses.append(test_loss)
        test_accs.append(test_acc)

        # Print epoch results
        print(f"\nEpoch {epoch+1}/{CFG['epochs']}")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2%}")
        print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2%}")
        
        # Classification reports
        print("\nTraining Classification Report:")
        print(classification_report(all_labels_train, all_preds_train, target_names=AudioEmotionModel.EMOTIONS))
        
        print("\nValidation Classification Report:")
        print(classification_report(all_labels_test, all_preds_test, target_names=AudioEmotionModel.EMOTIONS))

    # Save models
    torch.save(model.state_dict(), "./models/audio_emotion_model.pth")
    print("\nModel saved to ./models/audio_emotion_model.pth")

    # Convert to TorchScript
    model.eval()
    model_cpu = model.to("cpu")
    example_input = torch.rand(1, CFG["seq_length"], CFG["input_size"])
    traced_model = torch.jit.trace(model_cpu, example_input)
    traced_model.save("./models/audio_emotion_model.pt")
    print("TorchScript model saved to ./models/audio_emotion_model.pt")

    # Plot metrics
    plot_metrics(train_losses, test_losses, train_accs, test_accs)

    return model

def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for features, labels in loader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return (
        running_loss / len(loader),
        correct / len(loader.dataset),
        all_preds,
        all_labels
    )

def plot_metrics(train_loss, test_loss, train_acc, test_acc):
    plt.figure(figsize=(12, 5))
    
    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(train_loss, label='Train')
    plt.plot(test_loss, label='Test')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(train_acc, label='Train')
    plt.plot(test_acc, label='Test')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("./models/training_metrics.png")
    plt.show()

if __name__ == "__main__":
    train_model()