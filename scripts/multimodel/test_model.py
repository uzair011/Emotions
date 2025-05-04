import torch

try:
    model = torch.load("models/emotion_model.pth", map_location='cpu', weights_only=True)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Corrupted model file: {str(e)}")