import os
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

# Path to dataset
DATASET_PATH = "./datasets/"

# Output CSV file
FEATURES_CSV = "./datasets/audio_features.csv"

# all emotion labels in RAVDESS
emotion_mapping = {
    "01": "Neutral",  # 7 emotions in visual and 8 emotions in voice. So, calm and neutral are merged here
    "02": "Neutral",
    "03": "Happy",
    "04": "Sad",
    "05": "Angry",
    "06": "Fearful",
    "07": "Disgust",
    "08": "Surprise"
}

# Extract MFCC features
def extract_features(file_path):
    y, sr = librosa.load(file_path, duration=3, offset=0.5)  # Load audio
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)  # Extract clips
    return np.mean(mfcc.T, axis=0)  # Compute mean across time axis

# Process  audio files
features = []
for root, _, files in tqdm(os.walk(DATASET_PATH)):
    for file in files:
        if file.endswith(".wav"):
            emotion_code = file.split("-")[2]  # Extract emotion code label
            emotion = emotion_mapping.get(emotion_code, "Unknown")
            file_path = os.path.join(root, file)

            # Extract MFCC features
            mfcc_features = extract_features(file_path)

            # Append to list
            features.append([file_path, emotion] + list(mfcc_features))

# Convert to DataFrame & Save
df = pd.DataFrame(features, columns=["File", "Emotion"] + [f"MFCC_{i}" for i in range(13)])
df.to_csv(FEATURES_CSV, index=False)
print(f"✅ Audio features saved to {FEATURES_CSV}")
