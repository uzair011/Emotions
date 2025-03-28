import torch
import librosa
import numpy as np
import pyaudio
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import os
import soundfile as sf
from matplotlib.animation import FuncAnimation
from collections import deque
import time


# Load the trined model
class AudioEmotionModel(nn.Module):
    def __init__(self, input_size=13,
                hidden_size=128,
                num_layers=2,
                num_classes=7):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]) # select the last time stamp


# initialise the mosel
model = AudioEmotionModel()        
# load weights 
model.load_state_dict(torch.load(
    "./models/audio_emotion_model.pth",
    map_location = torch.device("cpu"),
    weights_only=True))
model.eval()

# CONSTANTS
# audio configuration
EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
SAMPLE_RATE = 16000
CHUNK_SIZE = 1024 # 64ms audio chunks
N_FFT = 1024
HOP_LENGTH = 512

# visualisation configuration
plt.style.use('ggplot')
fig, ax = plt.subplots(figsize=(14, 6))
emotion_lines = {emotion: ax.plot([], [], label=emotion, alpha=0.8)[0] for emotion in EMOTIONS}
ax.set_ylim(0, 100)
ax.set_xlabel('Time (seconds)')
ax.set_ylabel('Confidence (%)')
ax.legend()

# data buffers
HISTORY_SECONDS = 10
BUFFER_SIZE = int(SAMPLE_RATE / CHUNK_SIZE * HISTORY_SECONDS)
time_buffer = deque(maxlen=BUFFER_SIZE)
emotion_history = {e: deque(maxlen=BUFFER_SIZE) for e in EMOTIONS}


# Record real time audio
#!!! we no longer need this utility function
def record_audio(filename="real_time.wav", duration=10, sr=SAMPLE_RATE):
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=sr, input=True, frames_per_buffer=1024)
    
    print(f"RRrreccordinnngg!!! upto {duration} of seconds..")
    frames = []

    # calculate no of chunks needed. 
    total_chunks = int( sr/ 1024 * duration)

    try:
        for _ in range(total_chunks):
            data = stream.read(1024, exception_on_overflow=False)
            audio_frame = np.frombuffer(data, dtype=np.int16) # convert to numpy
            
            if audio_frame.size > 0:
                frames.append(audio_frame) 
            else:
                print("Warning! empty audio piece detected.")
    except Exception as e:
        print(f"Recording error: {str(e)}")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()

    if not frames:
        raise ValueError ("No audio data recorded.")    
    
    try: # explicit format specification
        sf.write(filename, np.concatenate(frames), sr, subtype="PCM_16")
        print(f"Sucessfully saved {len(frames)/sr:.1f}s audio to: {filename}")
    except Exception as e:
        print (f"file save error: {str(e)}")    
        raise

    
# CORE FUNCTIONS
def update_plot(frame):
    current_time = time_buffer[-1] if time_buffer else 0
    x_data = [t - current_time for t in time_buffer]
    
    for emotion in EMOTIONS:
        emotion_lines[emotion].set_data(x_data, emotion_history[emotion])
    ax.set_xlim(-HISTORY_SECONDS, 0)
    return list(emotion_lines.values())


def process_chunk(audio_chunk):
    try:
        if len(audio_chunk) < N_FFT:
            audio_chunk = np.pad(audio_chunk, (0, N_FFT - len(audio_chunk)))
            
        audio_float = audio_chunk.astype(np.float32) / np.iinfo(np.int16).max
        mfccs = librosa.feature.mfcc(
            y=audio_float,
            sr=SAMPLE_RATE,
            n_mfcc=13,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            center=False
        )
        return torch.tensor(mfccs.T[-1:], dtype=torch.float32).unsqueeze(0)
    except Exception as e:
        print(f"Processing error: {str(e)}")
        return None
    

# Predict Emotion
def predict_emotion(features):
    with torch.no_grad():
        outputs = model(features)
        probabilities = torch.softmax(outputs, dim=1).squeeze().numpy()
 
    return {e: float(probabilities[i] * 100) for i, e in enumerate(EMOTIONS)}


def main():

# initialise audio 
    p = pyaudio.PyAudio()
    stream = p.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=SAMPLE_RATE,
        input=True,
        frames_per_buffer=CHUNK_SIZE)

    # setup amimation
    anim = FuncAnimation(fig, 
                        update_plot,
                        blit=True,
                        interval= 50,
                        cache_frame_data=False,
                        save_count=BUFFER_SIZE)

    try:
        plt.show(block=False)
        print("Real-time emotion detection active. Ctrl+C to exit...")
        
        start_time =  time.time()

        while True:
            # Read audio chunk
            try:
                data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
                audio_chunk = np.frombuffer(data, dtype=np.int16)
                
                # Process chunk
                features = process_chunk(audio_chunk)
                if features is not None:  # Explicit None check
                    predictions = predict_emotion(features)
                    current_time = time.time() - start_time
                    
                    # Update buffers
                    time_buffer.append(current_time)
                    for emotion in EMOTIONS:
                        emotion_history[emotion].append(predictions[emotion])
                    
                    # Update display
                    fig.canvas.flush_events()
            
            except Exception as e:
                print(f"Runtime error: {str(e)}")
                continue
                
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()
        plt.close()

if __name__ == "__main__":
    main()   
