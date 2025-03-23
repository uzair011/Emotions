import torch
import torchaudio
import librosa
import numpy as np
import pyaudio
import wave
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim


# Load the trined model
class AudioEmotionModel(nn.Module):
    def __init__(self, input_size=13,
                hidden_size=64,
                num_layers=2,
                num_classes=7):
        super(AudioEmotionModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :]) # select the last time stamp
        return out

# initialise the mosel
model = AudioEmotionModel(input_size=13, hidden_size=64, num_classes=7)        

# load weights 
model.load_state_dict(torch.load(
    "./models/audio_emotion_model.pth",
    map_location = torch.device("cpu"),
    weights_only = True
    ))
model.eval()

# Record real time audio
def record_audio(filename="real_time..wav", duration=3, sr=16000):
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=sr, input=True, frames_per_buffer=1024)
    frames = []
    print("RRrreccordinnnggg!!!")

    for _ in range(0, int(sr / 1024 * duration)):
        data = stream.read(1024)
        frames.append(data)
    print("Done Recording!")

    stream.stop_stream()
    stream.close()
    p.terminate()
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
        wf.setframerate(sr)
        wf.writeframes(b"".join(frames))


# Extract MFCC features
def extract_features(filename, sr=16000):
    y, _ = librosa.load(filename, sr=sr)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return torch.tensor(mfccs.T, dtype=torch.float32).unsqueeze(0)  # Shape [1, time_steps, features]

EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
# Predict Emotion
def predict_emotion(features):
    with torch.no_grad():
        outputs = model(features)
        probabilities = torch.softmax(outputs, dim=1).squeeze().numpy()
    predictions = {EMOTIONS[i]: round(probabilities[i] * 100, 2) 
                   for i in range(len(EMOTIONS))}
    return predictions


# Plot waveform & spectrogram
def plot_audio(filename):
    y, sr = librosa.load(filename, sr=16000)
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(y)
    plt.title("Waveform")
    
    plt.subplot(1, 2, 2)
    spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
    librosa.display.specshow(librosa.power_to_db(spectrogram, ref=np.max), sr=sr)
    plt.title("Spectrogram")
    plt.colorbar()
    plt.show()

# Main function
def main():
    record_audio()
    features = extract_features("real_time.wav")
    predictions = predict_emotion(features)
    print("🎭 Emotion Predictions:", predictions)
    plot_audio("real_time.wav")

if __name__ == "__main__":
    main()

                        
        
