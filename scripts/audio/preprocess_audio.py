import os
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm
from scripts.audio.utils.audio_utils import extract_features, AudioEmotionModel

# Configuration
DATASET_PATH = "./datasets/"
FEATURES_CSV = "./datasets/audio_features.csv"
SEQ_LENGTH = 130  # Must match training sequence length
NUM_FEATURES = 39  # 13 MFCC + 13 delta + 13 delta-delta
EMOTION_MAP = {
    "01": "Neutral", "02": "Neutral", "03": "Happy",
    "04": "Sad", "05": "Angry", "06": "Fearful",
    "07": "Disgust", "08": "Surprise"
}

def process_sequence(mfccs, target_length):
    """Ensure fixed sequence length with padding/truncation"""
    if mfccs.shape[0] > target_length:
        return mfccs[:target_length]
    return np.pad(mfccs, ((0, target_length - mfccs.shape[0]), (0, 0)), mode='constant')

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
                # Process audio
                y, sr = librosa.load(file_path, duration=3, offset=0.5)
                mfccs = extract_features(y, sr)
                
                # Process sequence and flatten
                processed = process_sequence(mfccs, SEQ_LENGTH)
                features.append([file_path, emotion] + processed.flatten().tolist())
                
            except Exception as e:
                print(f"Error processing {file}: {str(e)}")
                continue
    
    # Create columns for 39 features × 130 time steps
    columns = ["File", "Emotion"] + \
        [f"FEAT_{f}_{t}" for t in range(SEQ_LENGTH) for f in range(NUM_FEATURES)]
    
    pd.DataFrame(features, columns=columns).to_csv(FEATURES_CSV, index=False)
    print(f"✅ Features saved to {FEATURES_CSV}")

if __name__ == "__main__":
    process_dataset()