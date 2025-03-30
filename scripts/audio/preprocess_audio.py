import os
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm
from scripts.audio.utils.audio_utils import extract_features  # Remove local version

# Configuration
DATASET_PATH = "./datasets/"
FEATURES_CSV = "./datasets/audio_features.csv"

# all emotion labels in RAVDESS
# 7 emotions in visual and 8 emotions in voice. So, calm and neutral are merged here
EMOTION_MAP = {
    "01": "Neutral", "02": "Neutral", "03": "Happy",
    "04": "Sad", "05": "Angry", "06": "Fearful",
    "07": "Disgust", "08": "Surprise"
}
SEQ_LENGTH = 130  # 3sec audio with 512 hop_length: (16000/512)*3 ≈ 93, rounded up

def process_dataset():
    features = []
    for root, _, files in tqdm(os.walk(DATASET_PATH)):
        for file in files:
            if not file.endswith(".wav"):
                continue
                
            parts = file.split("-")
            if len(parts) < 3 or parts[2] not in EMOTION_MAP:
                continue
                
            emotion = EMOTION_MAP[parts[2]]
            file_path = os.path.join(root, file)
            
            try:
                # Process audio once
                y, sr = librosa.load(file_path, duration=3, offset=0.5)
                mfccs = extract_features(y, sr)  # Should return (seq_len, 13)
                
                # Pad/truncate to fixed sequence length
                processed = process_sequence(mfccs, SEQ_LENGTH)
                features.append([file_path, emotion] + processed.flatten().tolist())
                
            except Exception as e:
                print(f"Error processing {file}: {str(e)}")
    
    # Create columns for fixed-length sequences
    columns = ["File", "Emotion"] + [f"MFCC_{i}_{t}" for t in range(SEQ_LENGTH) for i in range(13)]
    pd.DataFrame(features, columns=columns).to_csv(FEATURES_CSV, index=False)
    print(f"✅ Features saved to {FEATURES_CSV}")

def process_sequence(mfccs, target_length):
    """Ensure fixed sequence length with padding/truncation"""
    if mfccs.shape[0] > target_length:
        return mfccs[:target_length]
    return np.pad(mfccs, ((0, target_length - mfccs.shape[0]), (0, 0)), mode='constant')

if __name__ == "__main__":
    process_dataset()
