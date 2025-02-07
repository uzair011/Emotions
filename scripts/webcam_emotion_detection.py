import cv2
import torch
import torch.nn as nn
import numpy as np
from torchvision.models import resnet18
from torchvision import transforms
from PIL import Image

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

def load_model(model_path, device):
    model = resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, len(emotion_labels))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def preprocess_frame(frame):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((48, 48)),          # Resize to 48x48
        transforms.ToTensor(),                # Convert to tensor
        transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)  # Normalize pixel values for 3 channels
    ])
    return transform(frame).unsqueeze(0) # batch dimension


def predict_emotion(frame, model, device):
    image_tensor = preprocess_frame(frame).to(device)
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.nn.functional.softmax(output, dim = 1)

    # top 3
    top3_prob, top3_indices = torch.topk(probabilities, 3)
    top3_emotions = [emotion_labels[idx] for idx in top3_indices[0]]
    top3_confidences = [prob.item() * 100 for prob in top3_prob[0]]

    # output
    prediction_string = " | ".join([f"{emotion} ({confidence: .1f}%)" for emotion, confidence in zip(top3_emotions, top3_confidences)])    

    return prediction_string


def real_time_emotion_detection(model_path):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model_path, device)

    capture = cv2.VideoCapture(0) # select the default webcam

    while True:
        ret, frame = capture.read()
        if not ret:
            break # no frame capture, exit loop
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # convert bgr -> rgb 

        prediction = predict_emotion(rgb_frame, model, device)

        # display prediction on video
        cv2.putText(frame, prediction, [10, 40], cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

        # display webcam feed
        cv2.imshow("Real time Emotion Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"): # q for quit
            break

        # remove all resources
    capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    real_time_emotion_detection("./models/emotion_model.pth")