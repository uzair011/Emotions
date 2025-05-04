import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'  # Fix Metal compatibility
os.environ['OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS'] = '0'  # Disable hardware acceleration

import threading
import queue
import cv2
import pyaudio
import numpy as np
import torch
from torchvision import transforms
from ..audio.utils.audio_utils import extract_features
from ..visual.inference_resnet18 import load_model as load_visual_model 


class MultimodelEmotionDetector:
    def __init__(self):
        # Initialize models
        self.audio_model = torch.jit.load("models/audio_emotion_model.pt").eval()
        self.visual_model = load_visual_model("models/emotion_model.pth", torch.device('cpu'))
        
        # Buffers and queues with thread safety
        self.audio_buffer = np.zeros(16000 * 3)  # 3-second buffer
        self.video_queue = queue.Queue(maxsize=30)
        self.results = queue.Queue()
        self.latest_frame = None
        self.lock = threading.Lock()
        
        # Hardware setup with error handling
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("Could not open video device")
            
        self.audio = pyaudio.PyAudio()
        self.stream = self.audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=16000,
            input=True,
            frames_per_buffer=1024,
            start=False
        )

    def _audio_capture(self):
        try:
            while True:
                data = self.stream.read(1024, exception_on_overflow=False)
                audio = np.frombuffer(data, dtype=np.int16)
                with self.lock:
                    self.audio_buffer = np.roll(self.audio_buffer, -1024)
                    self.audio_buffer[-1024:] = audio
        except Exception as e:
            print(f"Audio capture error: {str(e)}")

    def _video_capture(self):
        try:
            while True:
                ret, frame = self.cap.read()
                if ret:
                    with self.lock:
                        self.latest_frame = frame.copy()
                        self.video_queue.put(cv2.resize(frame, (48, 48)))
        except Exception as e:
            print(f"Video capture error: {str(e)}")

    def _audio_process(self):
        while True:
            try:
                with self.lock:
                    audio = self.audio_buffer.copy()
                audio = audio.astype(np.float32) / 32768.0
                features = extract_features(audio, 16000)
                with torch.no_grad():
                    audio_probs = self.audio_model(torch.tensor(features).unsqueeze(0))
                self.results.put(('audio', audio_probs.numpy()))
            except Exception as e:
                print(f"Audio processing error: {str(e)}")

    def _visual_process(self):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        while True:
            try:
                frame = self.video_queue.get()
                tensor = transform(frame).unsqueeze(0)
                with torch.no_grad():
                    visual_probs = torch.softmax(self.visual_model(tensor), dim=1)
                self.results.put(('visual', visual_probs.numpy()))
            except Exception as e:
                print(f"Visual processing error: {str(e)}")

    def _fuse_results(self):
        audio_probs = None
        visual_probs = None
        
        while True:
            try:
                source, probs = self.results.get()
                
                if source == 'audio':
                    audio_probs = probs
                else:
                    visual_probs = probs
                
                if audio_probs is not None and visual_probs is not None:
                    combined = (audio_probs * 0.4) + (visual_probs * 0.6)
                    final_emotion = np.argmax(combined)
                    self._display_result(final_emotion)
                    audio_probs = visual_probs = None
            except Exception as e:
                print(f"Fusion error: {str(e)}")

    def _display_result(self, emotion_idx):
        try:
            EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
            with self.lock:
                if self.latest_frame is not None:
                    display_frame = self.latest_frame.copy()
                    cv2.putText(display_frame, f"Emotion: {EMOTIONS[emotion_idx]}", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    cv2.imshow("Multimodel Emotion Detection", display_frame)
                    cv2.waitKey(1)
        except Exception as e:
            print(f"Display error: {str(e)}")

    def start(self):
        # Start audio stream after thread creation
        self.stream.start_stream()
        
        threads = [
            threading.Thread(target=self._audio_capture),
            threading.Thread(target=self._video_capture),
            threading.Thread(target=self._audio_process),
            threading.Thread(target=self._visual_process),
            threading.Thread(target=self._fuse_results)
        ]
        
        for t in threads:
            t.daemon = True
            t.start()
        
        try:
            while True:
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            self._cleanup()

    def _cleanup(self):
        self.stream.stop_stream()
        self.stream.close()
        self.audio.terminate()
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    detector = MultimodelEmotionDetector()
    detector.start()