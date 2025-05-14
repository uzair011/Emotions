import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import platform
import os
import threading
import queue
import cv2
import pyaudio
import numpy as np
import torch
from torchvision import transforms

from scripts.audio.utils.audio_utils import extract_features
from scripts.visual.inference_resnet18 import load_model as load_visual_model 

if platform.system() == 'Darwin':
    cv2.ocl.setUseOpenCL(False)
    os.environ['OPENCV_VIDEOIO_PRIORITY_MSMF'] = '0'
    os.environ['OPENCV_VIDEOIO_PRIORITY_AVFOUNDATION'] = '1'

class MultimodelEmotionDetector:
    def __init__(self):
        print("[INFO] Initializing models...")
        self.audio_model = torch.jit.load("models/audio_emotion_model.pt").eval()
        self.visual_model = load_visual_model("models/emotion_model.pth", torch.device('cpu'))

        self.audio_buffer = np.zeros(16000 * 3)
        self.video_queue = queue.Queue(maxsize=30)
        self.results = queue.Queue()
        self.latest_frame = None
        self.lock = threading.Lock()

        print("[INFO] Accessing webcam...")
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("Could not open video device")

        print("[INFO] Setting up microphone...")
        self.audio = pyaudio.PyAudio()
        self.stream = self.audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=16000,
            input=True,
            frames_per_buffer=1024,
            start=False
        )
        self.running = True

    def _audio_capture(self):
        print("[THREAD] Audio capture started")
        try:
            self.stream.start_stream()
            while self.running:
                data = self.stream.read(1024, exception_on_overflow=False)
                audio = np.frombuffer(data, dtype=np.int16)
                with self.lock:
                    self.audio_buffer = np.roll(self.audio_buffer, -1024)
                    self.audio_buffer[-1024:] = audio
        except Exception as e:
            print(f"[ERROR] Audio capture: {str(e)}")

    def _video_capture(self):
        print("[THREAD] Video capture started")
        try:
            while self.running:
                ret, frame = self.cap.read()
                if ret:
                    with self.lock:
                        self.latest_frame = frame.copy()
                        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        self.video_queue.put(cv2.resize(rgb_frame, (48, 48)))
        except Exception as e:
            print(f"[ERROR] Video capture: {str(e)}")

    def _audio_process(self):
        print("[THREAD] Audio processing started")
        while self.running:
            try:
                with self.lock:
                    audio = self.audio_buffer.copy()
                audio = audio.astype(np.float32) / 32768.0
                features = extract_features(audio, 16000)
                with torch.no_grad():
                    audio_probs = self.audio_model(torch.tensor(features).unsqueeze(0))
                self.results.put(('audio', audio_probs.numpy()))
            except Exception as e:
                print(f"[ERROR] Audio processing: {str(e)}")

    def _visual_process(self):
        print("[THREAD] Visual processing started")
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        while self.running:
            try:
                frame = self.video_queue.get()
                tensor = transform(frame).unsqueeze(0)
                with torch.no_grad():
                    visual_probs = torch.softmax(self.visual_model(tensor), dim=1)
                self.results.put(('visual', visual_probs.numpy()))
            except Exception as e:
                print(f"[ERROR] Visual processing: {str(e)}")

    def _fuse_results(self):
        print("[THREAD] Fusion started")
        audio_probs = None
        visual_probs = None
        while self.running:
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
                print(f"[ERROR] Fusion: {str(e)}")

    def _display_result(self, emotion_idx):
        EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
        try:
            with self.lock:
                if self.latest_frame is not None:
                    display_frame = self.latest_frame.copy()
                    cv2.putText(display_frame, f"Emotion: {EMOTIONS[emotion_idx]}", 
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    self.latest_frame = display_frame
        except Exception as e:
            print(f"[ERROR] Display: {str(e)}")

    def start(self):
        print("[INFO] Starting emotion detection system...")
        print("[INFO] Press 'q' to quit.")

        # Start all threads
        threading.Thread(target=self._audio_capture, daemon=True).start()
        threading.Thread(target=self._video_capture, daemon=True).start()
        threading.Thread(target=self._audio_process, daemon=True).start()
        threading.Thread(target=self._visual_process, daemon=True).start()
        threading.Thread(target=self._fuse_results, daemon=True).start()

        try:
            while True:
                if self.latest_frame is not None:
                    frame = self.latest_frame.copy()
                    cv2.putText(frame, "Press 'q' to quit", (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                    cv2.imshow("Multimodel Emotion Detection", frame)
                else:
                    dummy = np.zeros((240, 320, 3), dtype=np.uint8)
                    cv2.putText(dummy, "Waiting for camera...", (10, 120),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
                    cv2.imshow("Multimodel Emotion Detection", dummy)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            self._cleanup()

    def _cleanup(self):
        print("[INFO] Cleaning up...")
        self.running = False
        if self.stream.is_active():
            self.stream.stop_stream()
        self.stream.close()
        self.audio.terminate()
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    detector = MultimodelEmotionDetector()
    detector.start()
