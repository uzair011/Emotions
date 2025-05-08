import librosa
import numpy as np
import torch
import torch.nn as nn

class AudioEmotionModel(nn.Module):
    EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    
    def __init__(self, input_dim=39, hidden_dim=256, num_layers=3, num_classes=7):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=True,  # bidirectional
            batch_first=True
        )
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)  # x2 for bidirectional

    def forward(self, x):
        # Input shape: (batch_size, sequence_length, input_size)
        lstm_out, _ = self.lstm(x)
        lstm_out = self.dropout(lstm_out[:, -1, :])  # Take last timestep
        return self.fc(lstm_out)

def extract_features(audio, sr):
    # Normalise audio first
    audio = librosa.util.normalize(audio)
    
    # Extract MFCCs with correct settings
    mfccs = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=13,
        n_fft=2048,
        hop_length=512,
        lifter=40
    )
    
    # Add delta and delta-delta features
    delta = librosa.feature.delta(mfccs)
    delta2 = librosa.feature.delta(mfccs, order=2)
    
    # Combine features (13 MFCC + 13 delta + 13 delta2 = 39 features)
    combined = np.concatenate([mfccs, delta, delta2], axis=0)
    
    return combined.T  # (time_steps, 39 features)