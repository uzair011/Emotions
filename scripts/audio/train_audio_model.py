# # Training script
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import pandas as pd
# import numpy as np
# from torch.utils.data import DataLoader, Dataset
# from sklearn.preprocessing import LabelEncoder
# from sklearn.model_selection import train_test_split
# from tqdm import tqdm
# import matplotlib.pyplot as plt

# # HYPERPARAMETERS
# BATCH_SIZE = 32
# EPOCHS = 34
# LEARNING_RATE = 0.001
# NUM_FEATURES = 13
# NUM_CLASSES = 7

# # load the dataset
# df = pd.read_csv("./datasets/audio_features.csv")

# # Add this immediately after loading the DataFrame
# print(f"DataFrame shape: {df.shape}")
# print("First 5 rows:\n", df.head())

# # labeling - emotions as numbers
# label_encoder = LabelEncoder()
# df["Emotion"] = label_encoder.fit_transform(df["Emotion"])

# # train and test split
# train_data, test_data = train_test_split(df, test_size=0.2, random_state=42, stratify=df["Emotion"])


# # custom dataset class
# class AudioDataset(Dataset):
#     def __init__(self, data):
#         self.features = data.iloc[:, 2:].values.astype(np.float32) # mfcc features
#         self.labels = data.iloc[:, 1].values # emotion labels


#     def __len__(self):
#         return len(self.features)
    
#     def __getitem__(self, idx):
#         return torch.tensor(self.features[idx]), torch.tensor(self.labels[idx], dtype=torch.long)
    

# # create data loaders
# train_dataset = AudioDataset(train_data)
# test_dataset = AudioDataset(test_data)
# train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# # define LSTM model for audio emotion recognition
# class AudioLSTM(nn.Module):
#     def __init__(self, input_size=NUM_FEATURES, hidden_size=128, num_layers=2, num_classes=NUM_CLASSES):
#         super(AudioLSTM, self).__init__()
#         self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
#         self.fc = nn.Linear(hidden_size, num_classes)

#     def forward(self, x):
#         x = x.unsqueeze(1)  # time dimension for LSTM
#         lstm_out, _ = self.lstm(x)
#         return self.fc(lstm_out[:, -1, :])  # return last LSTM timestep

            
# # model
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = AudioLSTM().to(device)
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# # store data for visualisation
# train_losses = []
# train_accuracies = []
# test_losses = []
# test_accuracies = []

# # train the model
# def train_model():
#     model.train()
#     for epoch in range(EPOCHS):
#         total_loss, correct = 0, 0

#         for features, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):    
#             features, labels = features.to(device), labels.to(device)

#             optimizer.zero_grad()
#             outputs = model(features)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()

#             total_loss += loss.item()
#             correct += (outputs.argmax(1) == labels).sum().item()

#         train_loss = total_loss/ len(train_loader)
#         train_acc = correct / len(train_dataset)    
#         train_losses.append(train_loss)  # Track TRAIN loss
#         train_accuracies.append(train_acc)

#         # evaluate on the test set
#         model.eval()
#         test_correct, test_loss = 0, 0

#         with torch.no_grad():
#             for features, labels in test_loader:
                
#                 features, labels = features.to(device), labels.to(device)
#                 outputs = model(features)
#                 loss = criterion(outputs, labels)

#                 test_loss += loss.item()
#                 test_correct += (outputs.argmax(1) == labels).sum().item()



#         test_loss = test_loss/ len(test_loader)
#         test_acc = test_correct/ len(test_dataset)
#         test_losses.append(test_loss)  # Track TEST loss
#         test_accuracies.append(test_acc)

#         # if train_losses and train_accuracies and test_losses and test_accuracies:
#         #     print(f"Epoch {epoch+1}: Train Loss={train_losses[-1]:.4f}, Train Acc={train_accuracies[-1]:.4f} | Test Loss={test_losses[-1]:.4f}, Test Acc={test_accuracies[-1]:.4f}")

#         print(f"Epoch {epoch+1}: "
#               f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f} | "
#               f"Test Loss={test_loss:.4f}, Test Acc={test_acc:.4f}")


# #  visualize training results
# def plot_metrics():
#     epochs_range = range(1, EPOCHS + 1)

#     plt.figure(figsize=(12, 5))

#     # Loss Plot
#     plt.subplot(1, 2, 1)
#     plt.plot(epochs_range, train_losses, label='Train Loss', marker='o')
#     plt.plot(epochs_range, test_losses, label='Test Loss', marker='o')
#     plt.title("Loss Over Epochs")
#     plt.xlabel("Epochs")
#     plt.ylabel("Loss")
#     plt.legend()

#     # Accuracy Plot
#     plt.subplot(1, 2, 2)
#     plt.plot(epochs_range, train_accuracies, label='Train Accuracy', marker='o')
#     plt.plot(epochs_range, test_accuracies, label='Test Accuracy', marker='o')
#     plt.title("Accuracy Over Epochs")
#     plt.xlabel("Epochs")
#     plt.ylabel("Accuracy")
#     plt.legend()

#     plt.tight_layout()
#     plt.show()

# # Train Model
# train_model()

# # Debugging:
# print(f"Train Losses: {train_losses}")
# print(f"Test Losses: {test_losses}")

# if not train_losses or not test_losses:
#     print("Error: Train/Test loss lists are empty. Check the training loop!")
#     exit()  # Stop script if data is missing

# plot_metrics()


# # Save Trained Model
# torch.save(model.state_dict(), "./models/audio_emotion_model.pth")
# print("Model saved to ./models/audio_emotion_model.pth")

### ================

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from scripts.audio.utils.audio_utils import AudioEmotionModel  # Updated model
from scripts.audio.utils.data_loader import get_data_loaders

# Configuration
CFG = {
    "batch_size": 32,
    "epochs": 34,
    "lr": 0.001,
    "input_size": 13,
    "hidden_size": 128,
    "num_layers": 2,
    "num_classes": 7
}

def train_model():
    # Initialise
    train_loader, test_loader, _ = get_data_loaders("./datasets/audio_features.csv", CFG["batch_size"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = AudioEmotionModel(
        CFG["input_size"], 
        CFG["hidden_size"],
        CFG["num_layers"],
        CFG["num_classes"]
    ).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=CFG["lr"])
    
    # Training loop
    for epoch in range(CFG["epochs"]):
        model.train()
        epoch_loss, correct = 0, 0
        
        for features, labels in train_loader:
            print("Input shape:", features.shape)  # Should be (batch, seq_len, 13)
            features = features.to(device)
            outputs = model(features)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(features)
            loss = criterion(outputs, labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Track metrics
            epoch_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()
        
        # Validation
        train_loss = epoch_loss / len(train_loader)
        train_acc = correct / len(train_loader.dataset)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        
        print(f"Epoch {epoch+1}/{CFG['epochs']}")
        print(f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.2%}")
        print(f"Test Loss: {test_loss:.4f} | Acc: {test_acc:.2%}\n")


def evaluate(model, loader, criterion, device):
    model.eval()
    loss, correct = 0, 0
    
    with torch.no_grad():
        for features, labels in loader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            loss += criterion(outputs, labels).item()
            correct += (outputs.argmax(1) == labels).sum().item()
    
    return loss/len(loader), correct/len(loader.dataset)

def plot_metrics(train_loss, test_loss, train_acc, test_acc):
    plt.figure(figsize=(12,5))
    
    plt.subplot(1,2,1)
    plt.plot(train_loss, label='Train')
    plt.plot(test_loss, label='Test')
    plt.title('Loss')
    plt.legend()
    
    plt.subplot(1,2,2)
    plt.plot(train_acc, label='Train')
    plt.plot(test_acc, label='Test')
    plt.title('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    train_model()