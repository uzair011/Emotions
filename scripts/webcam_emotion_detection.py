import cv2
import torch
import torch.nn as nn
import numpy as np
from torchvision.models import resnet18
from torchvision import transforms
from PIL import Image


# load haar cascade for face detection
FACE_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
FACE_CASCADE = cv2.CascadeClassifier(FACE_CASCADE_PATH)

if FACE_CASCADE.empty():
    raise FileNotFoundError(f"Could not load Haar cascade from {FACE_CASCADE_PATH}")

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# load traied model
def load_model(model_path, device):
    model = resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, len(emotion_labels))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

# preprocess a SINGLE face screenshot
def preprocess_frame(frame):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((48, 48)),          # Resize to 48x48
        transforms.ToTensor(),                # Convert to tensor
        transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)  # Normalize pixel values for 3 channels
    ])
    return transform(frame).unsqueeze(0) # batch dimension


def predict_emotion(face, model, device):
    face_image_tensor = preprocess_frame(face).to(device)
    with torch.no_grad():
        output = model(face_image_tensor)
        probabilities = torch.nn.functional.softmax(output, dim = 1)

    # top 3
    top3_prob, top3_indices = torch.topk(probabilities, 3)
    top3_emotions = [emotion_labels[idx] for idx in top3_indices[0]]
    top3_confidences = [prob.item() * 100 for prob in top3_prob[0]]

    # output
    prediction_string = " | ".join([f"{emotion} ({confidence: .1f}%)" for emotion, confidence in zip(top3_emotions, top3_confidences)])    

    return prediction_string

# real time webcam emotion and face detection
def real_time_emotion_detection(model_path):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model_path, device)

    capture = cv2.VideoCapture(0) # select the default webcam

    while True:
        ret, frame = capture.read()
        if not ret:
            break # no frame capture, exit loop
        
        # convert frame to grayscale (haar - grayscale images)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # convert bgr -> rgb 

        # DETECT FACE
        faces = FACE_CASCADE.detectMultiScale(gray, scaleFactor= 1.3, minNeighbors= 5, minSize= (50, 50))
        #prediction = predict_emotion(rgb_frame, model, device)

        for (x, y, w, h) in faces:
            face_roi = frame [y: y+h, x: x+w] # to crop the face area

            # predict emotion
            prediction = predict_emotion(face_roi, model, device)    

            # draw a box 
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # display predicted emotion text
            cv2.putText(frame, prediction, [10, 40], cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)


        # display webcam feed
        cv2.imshow("Real time Emotion Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"): # q for quit
            break

        # remove all resources
    capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    real_time_emotion_detection("./models/emotion_model.pth")