 # Audio processing functions

import librosa
import numpy as np
import torch
import torch.nn as nn


class AudioEmotionModel(nn.Module):
    def __init__(self, input_size=13, hidden_size=128, num_layers=2, num_classes=7):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # Input shape: (batch_size, sequence_length, input_size)
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])  # Take last timestep output

def extract_features(y, sr):
    """Should be in audio_utils.py"""
    mfccs = librosa.feature.mfcc(
        y=y, sr=sr, 
        n_mfcc=13, 
        n_fft=2048, 
        hop_length=512,
        n_mels=40
    )
    return mfccs.T  # (time_steps, 13)


# def extract_features(audio, sr=16000, n_mfcc=13, n_fft=1024, hop_length=512):
#     """Process audio chunk for real-time emotion prediction"""
#     if len(audio) < n_fft:
#         audio = np.pad(audio, (0, n_fft - len(audio)))
    
#     audio_float = audio.astype(np.float32) / np.iinfo(np.int16).max
#     mfccs = librosa.feature.mfcc(
#         y=audio_float,
#         sr=sr,
#         n_mfcc=n_mfcc,
#         n_fft=n_fft,
#         hop_length=hop_length,
#         center=False
#     )
#     return torch.tensor(mfccs.T[-1:], dtype=torch.float32).unsqueeze(0)    