# 🎯 emotion-aware-core

A modular framework for **real-time emotion recognition** via facial expressions and speech input.

---

## 📚 Table of Contents

- [About](#-about)
- [Features](#-features)
- [Installation](#-installation)
- [Usage](#-usage)
- [Customization](#-customization)
- [Contributing](#-contributing)
- [License](#-license)
- [Contact](#-contact)
- [Links](#-links)

---

## 🔍 About

`emotion-aware-core` is an open-source, modular framework designed to detect human emotions in real-time using **multimodal inputs**—facial expressions and speech. This system acts as an emotional intelligence layer, adaptable across various domains:

- 🎧 **Customer service** — escalate calls when frustration is detected
- 🧠 **Mental health apps** — monitor long-term mood changes
- 🛡️ **Security systems** — flag suspicious emotional behavior
- 📚 **Education tools** — track engagement levels during learning

**Key Philosophy**:
- 🧩 **Modular**: Swap models and detectors without altering the pipeline
- ⚡ **Real-Time**: Optimized for consumer-grade CPUs/GPUs (~12 FPS)
- 🧪 **Ethical AI**: Built with bias-awareness and transparency in mind

**Tech Stack**:
- 🎥 Visual: `ResNet-18` (FER2013) + `Haar Cascades`
- 🎙️ Audio: `1D-CNN` (RAVDESS) + `Librosa`
- 🔀 Fusion: Weighted averaging (Visual 0.6 / Audio 0.4)

---

## 🚀 Features

- ⏱️ Real-time emotion detection from webcam and microphone
- 🔄 Modular design for easy integration and upgrades
- 🛠️ Prebuilt demos: CLI, webcam inference, and REST API
- 💻 Cross-platform support (Linux, macOS, Windows)

---

## 📥 Installation

### Prerequisites
- Python 3.8+
- pip

### Steps

**1. Clone the repository**
```bash
git clone https://github.com/your-username/emotion-aware-core.git
cd emotion-aware-core
```

2. Install dependencies
`pip install -r requirements.txt`

3. Download pretrained models
 ```
# Download ResNet-18 and 1D-CNN model weights
wget https://your-model-hosting.com/models/resnet18_weights.pth -P models/
wget https://your-model-hosting.com/models/1dcnn_weights.h5 -P models/
```

## 🖥️ Usage
1. Real-Time Webcam Demo
`python scripts/multimodel/realtime_inference.py`

Output:

- Webcam window with predicted emotion overlays
- Terminal logs showing emotion scores and frame latency

## 2. CLI Tools
Image Inference:  
`python scripts/visual/inference_resnet18.py --image path/to/image.jpg`

Audio Inference:
`python scripts/audio/inference_1dcnn.py --audio path/to/audio.wav`

## 3. API Integration
```
from core.emotion_pipeline import EmotionEngine

engine = EmotionEngine()
emotion = engine.predict(frame, audio_clip)
```

## 🤝 Contributing
We welcome your ideas and code! Here’s how to get started:

- Fork the repository
- Create a branch

`git checkout -b feature/your-feature`

- Commit changes
`git commit -m "Add your feature"`

- Push and open a pull request

## 📜 License
This project is licensed under the MIT License.

## 🔗 Links
FER2013 Dataset => https://www.kaggle.com/datasets/msambare/fer2013
RAVDESS Dataset => https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio
