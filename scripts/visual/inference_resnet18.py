# import torch
# import torch.nn as nn
# from torchvision.models import resnet18
# from torchvision.models import ResNet18_Weights
# from torchvision import transforms
# from PIL import Image
# import matplotlib.pyplot as plt
# import argparse

# # Define the Model Class (same as in training)
# class EmotionResNet18(nn.Module):
#     def __init__(self, num_classes=7):
#         super(EmotionResNet18, self).__init__()
#         self.model = resnet18(pretrained=False)
#         self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

#     def forward(self, x):
#         return self.model(x)

# # Emotion Labels
# emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# # Load the Pre-trained Model
# def load_model(model_path, device):
#     model = resnet18(weights=None)
#     model.fc = torch.nn.Linear(model.fc.in_features, 7)
# #   model.load_state_dict(torch.load(model_path, map_location=device))
#     model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
#     model.eval()
#     return model

# # Preprocess the Input Image
# def preprocess_image(image_path):
#     transform = transforms.Compose([
#         transforms.Resize((48, 48)),          # Resize to 48x48
#         transforms.ToTensor(),                # Convert to tensor
#         transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)  # Normalize pixel values for 3 channels
#     ])

#     image = Image.open(image_path).convert('RGB')  # Open image
#     return transform(image).unsqueeze(0)           # Add batch dimension

# # Predict top 3 Emotions with score
# def predict_emotion(image_path, model, device):
#     image_tensor = preprocess_image(image_path).to(device)
#     with torch.no_grad():
#         output = model(image_tensor)
#         probabilities = torch.nn.functional.softmax(output, dim = 1)

#         #confidence, predicted_class = torch.max(probabilities, dim = 1)

#     # top 3
#     top3_prob, top3_indices = torch.topk(probabilities, 3)
#     top3_emotions = [emotion_labels[idx] for idx in top3_indices[0]]
#     top3_confidences = [prob.item() * 100 for prob in top3_prob[0]]

#     # output
#     prediction_string = " | ".join([f"{emotion} ({confidence: .2f}%)" for emotion, confidence in zip(top3_emotions, top3_confidences)])    

#     return prediction_string


# def visualise_prediction(image_path, predicted_emotion): ### Here
#     image = Image.open(image_path)
#     plt.imshow(image)
#     plt.axis('off') # hide axes
#     plt.title(f"Predicted: {predicted_emotion}", fontsize=16, color="red") ### here
#     plt.show()

# # Main Function for Inference
# def main():
#     import argparse
#     parser = argparse.ArgumentParser(description="Inference for Emotion Detection")
#     parser.add_argument('--image', type=str, required=True, help="Path to the input image")
#     parser.add_argument('--model', type=str, default='./models/emotion_model.pth', help="Path to the trained model")
#     args = parser.parse_args()

#     # Load the model
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     model = load_model(args.model, device)

#     # Perform inference
#     emotion = predict_emotion(args.image, model, device)
#     print(f"Predicted Emotion: {emotion}")

#     # visualise
#     visualise_prediction(args.image, emotion)


# if __name__ == "__main__":
#     main()

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt
import argparse
import os

# Define the Model Class (must match training exactly)
class EmotionResNet18(nn.Module):
    def __init__(self, num_classes=7):
        super(EmotionResNet18, self).__init__()
        self.base_model = models.resnet18(pretrained=False)
        in_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.base_model(x)

# Emotion Labels (verify order matches training)
EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

def load_model(model_path, device):
    """Load model with proper error handling"""
    try:
        model = EmotionResNet18(num_classes=7)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        return model.to(device)
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {str(e)}")

def preprocess_image(image_path):
    """Robust image preprocessing with error handling"""
    try:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found at {image_path}")
            
        image = Image.open(image_path).convert('RGB')
        
        transform = transforms.Compose([
            transforms.Resize((48, 48)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet stats
                                 std=[0.229, 0.224, 0.225])
        ])
        
        return transform(image).unsqueeze(0)
        
    except Exception as e:
        raise RuntimeError(f"Image processing failed: {str(e)}")

def predict_emotion(image_path, model, device):
    """Safe prediction with top-3 results"""
    try:
        image_tensor = preprocess_image(image_path).to(device)
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            top3_prob, top3_indices = torch.topk(probabilities, 3)
            
        top3_emotions = [EMOTION_LABELS[idx] for idx in top3_indices[0].cpu().numpy()]
        top3_confidences = [prob.item() * 100 for prob in top3_prob[0]]
        
        return " | ".join(
            [f"{emotion} ({confidence:.2f}%)" 
             for emotion, confidence in zip(top3_emotions, top3_confidences)]
        )
    except Exception as e:
        return f"Prediction error: {str(e)}"

def visualize_prediction(image_path, prediction):
    """Safe visualization"""
    try:
        image = Image.open(image_path)
        plt.imshow(image)
        plt.axis('off')
        plt.title(f"Predicted: {prediction}", fontsize=12, color='darkred')
        plt.show()
    except Exception as e:
        print(f"Visualization failed: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description="Emotion Detection Inference")
    parser.add_argument('--image', type=str, required=True,
                      help="Path to input image file")
    parser.add_argument('--model', type=str, default='models/emotion_model.pth',
                      help="Path to trained model weights")
    args = parser.parse_args()

    # Convert to absolute paths
    image_path = os.path.abspath(args.image)
    model_path = os.path.abspath(args.model)

    # Verify paths
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}")
        return
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        model = load_model(model_path, device)
        prediction = predict_emotion(image_path, model, device)
        print(f"Prediction Results: {prediction}")
        visualize_prediction(image_path, prediction)
    except Exception as e:
        print(f"Error during inference: {str(e)}")

if __name__ == "__main__":
    main()