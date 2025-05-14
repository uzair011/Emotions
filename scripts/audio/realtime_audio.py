import pyaudio
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import deque
import time
import threading
import librosa
from threading import Lock
from scripts.audio.utils.audio_utils import extract_features
import os

class AudioEmotionDetector:
    EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    
    def __init__(self):
        # Model configuration
        self.model = torch.jit.load("./models/audio_emotion_model.pt", map_location='cpu')
        self.model.eval()
        
        # Audio configuration
        self.sr = 16000          # Sample rate
        self.chunk_size = 512    # Matches MFCC hop_length
        self.buffer_duration = 3  # Seconds
        self.audio_buffer = np.zeros(self.sr * self.buffer_duration)
        self.last_process_time = time.time()

        # Visualization configuration
        self.buffer_size = 100    # Number of points to display
        self.history_seconds = 10 # Time window for visualization
        
        # Data buffers with thread lock
        self.buffer_lock = Lock()
        self.time_buffer = deque(maxlen=self.buffer_size)
        self.emotion_buffers = {e: deque(maxlen=self.buffer_size) for e in self.EMOTIONS}
        
        # Control flags
        self.running = False
        
        # Initialize plot
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(12, 4))
        self.lines = {e: self.ax.plot([], [], label=e)[0] for e in self.EMOTIONS}
        self.ax.set_ylim(0, 100)
        self.ax.legend(loc='upper right')
        self.ax.set_xlabel('Time (seconds)')
        self.ax.set_ylabel('Probability (%)')
        
        # Audio stream objects
        self.p = None
        self.stream = None

    def start(self):
        # Initialize audio stream
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.sr,
            input=True,
            frames_per_buffer=self.chunk_size,
            start=True
        )
        
        # Start processing thread
        self.running = True
        self.process_thread = threading.Thread(target=self.process_audio, daemon=True)
        self.process_thread.start()
        
        # Start animation
        self.ani = FuncAnimation(
            self.fig,
            self.update_plot,
            blit=True,
            interval=50,
            cache_frame_data=False,
            save_count=100
        )
        
        # Start main GUI loop
        try:
            plt.show(block=True)
        except KeyboardInterrupt:
            self.stop()

    def process_audio(self):
        try:
            while self.running:
                try:
                    # Read audio chunk
                    raw_data = self.stream.read(
                        self.chunk_size,
                        exception_on_overflow=False
                    )
                    audio = np.frombuffer(raw_data, dtype=np.int16)
                    audio_float = audio.astype(np.float32) / 32768.0

                    # Update rolling buffer
                    self.audio_buffer = np.roll(self.audio_buffer, -self.chunk_size)
                    self.audio_buffer[-self.chunk_size:] = audio_float
                    
                    # Process every 3 seconds
                    current_time = time.time()
                    if current_time - self.last_process_time >= self.buffer_duration:
                        features = extract_features(self.audio_buffer, self.sr)
                        processed = self.process_sequence(features, 130)
                        
                        with torch.no_grad():
                            input_tensor = torch.tensor(processed).float().unsqueeze(0)
                            outputs = self.model(input_tensor)
                            probs = torch.softmax(outputs, dim=1).squeeze().numpy()

                        # Update buffers with lock
                        with self.buffer_lock:
                            self.time_buffer.append(current_time)
                            for i, e in enumerate(self.EMOTIONS):
                                self.emotion_buffers[e].append(probs[i] * 100)
                        
                        self.last_process_time = current_time
                        print(f"Predictions: {dict(zip(self.EMOTIONS, probs))}")

                except OSError as e:
                    if e.errno == -9981:  # Input overflow
                        print("Buffer overflow - resetting stream...")
                        self.stream.stop_stream()
                        self.stream.start_stream()
                        
        except Exception as e:
            print(f"Processing error: {str(e)}")
        finally:
            self.stop()

    @staticmethod
    def process_sequence(mfccs, target_length):
        """Ensure fixed sequence length with padding/truncation"""
        if mfccs.shape[0] > target_length:
            return mfccs[:target_length]
        return np.pad(mfccs, ((0, target_length - mfccs.shape[0]), (0, 0)), mode='constant')

    def update_plot(self, frame):
        with self.buffer_lock:
            if self.time_buffer:
                current_time = self.time_buffer[-1]
                x_data = [t - current_time for t in self.time_buffer]
                
                # Convert numpy values to regular floats
                for emotion in self.EMOTIONS:
                    y_data = [float(v) for v in self.emotion_buffers[emotion]]
                    self.lines[emotion].set_data(x_data, y_data)
                
                self.ax.set_xlim(-self.history_seconds, 0)
                self.ax.relim()
                self.ax.autoscale_view(scaley=False)
        
        return list(self.lines.values())

    def stop(self):
        if not self.running:
            return
        
        self.running = False
        
        # Stop animation
        if hasattr(self, 'ani'):
            self.ani.event_source.stop()
        
        # Close plot
        plt.close(self.fig)
        
        # Cleanup audio resources
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        if self.p:
            self.p.terminate()
        
        # Wait for thread
        if self.process_thread.is_alive():
            self.process_thread.join(timeout=1)
        
        print("System fully stopped")

    @classmethod
    def test_audio_file(cls, file_path, model_path):
        """Test audio file processing"""
        model = torch.jit.load(model_path, map_location='cpu')
        model.eval()
        
        y, sr = librosa.load(file_path, duration=3)
        features = extract_features(y, sr)
        processed = cls.process_sequence(features, 130)

        print("MFCC Stats - Mean:", np.mean(features), "Std:", np.std(features), "Shape:", features.shape)
        
        with torch.no_grad():
            input_tensor = torch.tensor(processed).float().unsqueeze(0)
            outputs = model(input_tensor)
            probs = torch.softmax(outputs, dim=1).squeeze().numpy()
        
        print("Test predictions:", dict(zip(cls.EMOTIONS, probs)))




if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', type=str, help='Test audio file path')
    args = parser.parse_args()

    if args.test:
        AudioEmotionDetector.test_audio_file(args.test, "./models/audio_emotion_model.pt")
    else:
        detector = AudioEmotionDetector()
        try:
            detector.start()
        except KeyboardInterrupt:
            detector.stop()


