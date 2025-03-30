 # Shared data loading

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torch


SEQ_LENGTH = 130

class AudioDataset(Dataset):
    def __init__(self, data):
        self.features = data.iloc[:, 2:].values.astype(np.float32)
        self.labels = data.iloc[:, 1].values

    def __len__(self):
        return len(self.features)
    
    
    def __getitem__(self, idx):
        # Features are stored as flat arrays in CSV
        flat_features = self.features[idx]
        # Reshape to (sequence_length, input_size)
        features = flat_features.reshape(-1, 13)  # -1 infers sequence length
        return torch.tensor(features, dtype=torch.float32), torch.tensor(self.labels[idx])

def get_data_loaders(csv_path, batch_size=32, test_size=0.2):
    df = pd.read_csv(csv_path)
    le = LabelEncoder()
    df["Emotion"] = le.fit_transform(df["Emotion"])
    
    train_data, test_data = train_test_split(df, test_size=test_size, random_state=42, stratify=df["Emotion"])
    
    train_loader = DataLoader(AudioDataset(train_data), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(AudioDataset(test_data), batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader, le