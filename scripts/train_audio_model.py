import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# HYPERPARAMETERS
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 0.001
NUM_FEATURES = 13
NUM_CLASSES = 7

# load the dataset
df = pd.read_csv("./datasets/audio_features.csv")

# Add this immediately after loading the DataFrame
print(f"DataFrame shape: {df.shape}")
print("First 5 rows:\n", df.head())

# labeling - emotions as numbers
label_encoder = LabelEncoder()
df["Emotion"] = label_encoder.fit_transform(df["Emotion"])

# train and test split
train_data, test_data = train_test_split(df, test_size=0.2, random_state=42, stratify=df["Emotion"])


# custom dataset class
class AudioDataset(Dataset):
    def __init__(self, data):
        self.features = data.iloc[:, 2:].values.astype(np.float32) # mfcc features
        self.labels = data.iloc[:, 1].values # emotion labels


    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return torch.tensor(self.features[idx]), torch.tensor(self.labels[idx], dtype=torch.long)
    

# create data loaders
train_dataset = AudioDataset(train_data)
test_dataset = AudioDataset(test_data)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# define LSTM model for audio emotion recognition
class AudioLSTM(nn.Module):
    def __init__(self, input_size=NUM_FEATURES, hidden_size=64, num_layers=2, num_classes=NUM_CLASSES):
        super(AudioLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)  # time dimension for LSTM
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])  # return last LSTM timestep

            
# model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AudioLSTM().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# store data for visualisation
train_losses = []
train_accuracies = []
test_losses = []
test_accuracies = []

# train the model
def train_model():
    model.train()
    for epoch in range(EPOCHS):
        total_loss, correct = 0, 0

        for features, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            features, labels = features.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()
        
        #acc = correct / len(train_dataset)
        #print(f"Epoch {epoch+1}: Loss={total_loss/len(train_loader):.4f}, Accuracy={acc:.4f}")

        test_losses.append(total_loss / len(train_loader))
        train_accuracies.append(correct / len(train_dataset))

        # evaluate on the test set
        


# Train Model
train_model()

# Save Trained Model
torch.save(model.state_dict(), "./models/audio_emotion_model.pth")
print("Model saved to ./models/audio_emotion_model.pth")