import torch
import torch.nn as nn
from torchvision.models import resnet18
from torchvision import transforms
from PIL import Image

# Define the Model Class (same as in training)
class EmotionResNet18(nn.Module):
    def __init__(self, num_classes=7):
        super(EmotionResNet18, self).__init__()
        self.model = resnet18(pretrained=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

# Emotion Labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Load the Pre-trained Model
def load_model(model_path, device):
    model = EmotionResNet18(num_classes=7).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

# Preprocess the Input Image
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Grayscale(),               # FER2013 images are grayscale
        transforms.Resize((48, 48)),          # Resize to 48x48
        transforms.ToTensor(),                # Convert to tensor
        transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize pixel values
    ])

    image = Image.open(image_path).convert('RGB')  # Open image
    return transform(image).unsqueeze(0)           # Add batch dimension

# Predict Emotion
def predict_emotion(image_path, model, device):
    image_tensor = preprocess_image(image_path).to(device)
    with torch.no_grad():
        output = model(image_tensor)
        predicted_class = torch.argmax(output, dim=1).item()
    return emotion_labels[predicted_class]

# Main Function for Inference
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Inference for Emotion Detection")
    parser.add_argument('--image', type=str, required=True, help="Path to the input image")
    parser.add_argument('--model', type=str, default='./models/emotion_model.pth', help="Path to the trained model")
    args = parser.parse_args()

    # Load the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(args.model, device)

    # Perform inference
    emotion = predict_emotion(args.image, model, device)
    print(f"Predicted Emotion: {emotion}")

if __name__ == "__main__":
    main()
