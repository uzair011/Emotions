import sys
import os

# Project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from torchvision.models import resnet18
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.models import resnet18
from torchvision import transforms, datasets
from tqdm import tqdm
import matplotlib.pyplot as plt

# Import the FER2013Dataset and transform utilities
from utils.visual_data_loader import FER2013Dataset, get_transforms

# Define the Model
def get_model(num_classes=7):
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

# Training Function
def train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs=10):

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(epochs):
        # Training Phase
        model.train()
        train_loss = 0
        train_correct = 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} - Training"):
            images, labels = images.to(device), labels.to(device)

            # Forward Pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward Pass and Optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_correct += (outputs.argmax(1) == labels).sum().item()

        # Validation Phase
        model.eval()
        val_loss = 0
        val_correct = 0
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch + 1}/{epochs} - Validation"):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                val_correct += (outputs.argmax(1) == labels).sum().item()

        # store metrics
        train_losses.append(train_loss / len(train_loader))
        val_losses.append(val_loss / len(val_loader))
        train_accuracies.append(train_correct / len(train_loader.dataset))
        val_accuracies.append(val_correct / len(val_loader.dataset))

        # Print Epoch Summary
        # print(f"Epoch [{epoch + 1}/{epochs}] "
        #       f"Train Loss: {train_loss / len(train_loader):.4f}, "
        #       f"Train Acc: {train_correct / len(train_loader.dataset):.4f}, "
        #       f"Val Loss: {val_loss / len(val_loader):.4f}, "
        #       f"Val Acc: {val_correct / len(val_loader.dataset):.4f}")
        
        print(f"Epoch [{epoch + 1}/{epochs}] "
              f"Train Loss: {train_losses[-1]:.4f}, "
              f"Train Acc: {train_accuracies[-1]:.4f}, "
              f"Val Loss: {val_losses[-1]:.4f}, "
              f"Val Acc: {val_accuracies[-1]:.4f}")

    # plot metrics    
    plot_metrics(epochs, train_losses, val_losses, train_accuracies, val_accuracies)

# plot accuracy and loss
def plot_metrics(epochs, train_losses, val_losses, train_accuracies, val_accuracies):
    epochs_range = range(1, epochs + 1)

    plt.figure(figsize=(12, 5))
    
    # Loss Plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_losses, label='Train Loss')
    plt.plot(epochs_range, val_losses, label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Accuracy Plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, train_accuracies, label='Train Accuracy')
    plt.plot(epochs_range, val_accuracies, label='Validation Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()


def main():
    # Paths to CSV files
    train_folder = './datasets/train' 
    val_folder = './datasets/test/' 

    # transform images
    transform = transforms.Compose([
        transforms.Resize((48, 48)), # resize images to 48x48 
        transforms.ToTensor(), # convert images to pytorch tensors
        transforms.Normalize(mean = [0.5], std = [0.5]) # normalise pixel values
    ])

    # Load Datasets 
    train_dataset = datasets.ImageFolder(train_folder, transform=transform)
    val_dataset = datasets.ImageFolder(val_folder, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    # Device Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize Model, Loss, and Optimizer
    model = get_model(num_classes=7).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the Model
    train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs=10)

    # Save the Trained Model
    os.makedirs('./models', exist_ok=True)
    model_path = './models/emotion_model.pth'
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    main()
