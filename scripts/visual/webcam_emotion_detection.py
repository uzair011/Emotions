import cv2
import torch
import torch.nn as nn
import numpy as np
import csv
import os
import datetime
import threading
from torchvision.models import resnet18
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque

import matplotlib
matplotlib.use("TkAgg")  #  Force Matplotlib- working backend


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
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
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


def log_emotion(timestamp, emotions, confidences, log_file="emotions_log.csv"):
    """Log emotions to a csv file"""
    file_esists = os.path.isfile(log_file)

    with open(log_file, mode="a", newline="") as f:
        writer = csv.writer(f)

        if not file_esists: # write header if file is new
            header = ["Timestamp", "Emotion1", "Confidence1","Emotion2", "Confidence2","Emotion3", "Confidence3"]
            writer.writerow(header)

        row = [timestamp]  # inserting data to the rows  
        for emo, conf in zip(emotions, confidences):
            row += [emo, f"{conf: .1f}%"]

        writer.writerow(row)    


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

    return (prediction_string, top3_emotions, top3_confidences, top3_emotions[0])

# live data storage for live graphs
#time_window = deque(["00:00:00", "00:00:01"], maxlen=30)  # store last 30 timestamps
time_window = deque(maxlen=30)
emotion_counts = {emotion: 0 for emotion in emotion_labels} # count every emotions


def update_graphs(frame):
    plt.clf() # clear older frame data  
    
    if not time_window:
        return

    # filter emotions (ignore 0)
    filtered_emotions = {k: v for k, v in emotion_counts.items() if v > 0}

    if not filtered_emotions:
        return

    # line graph (emotion over time)
    # plt.subplot(1, 2, 1)
    # plt.plot(list(time_window), list(filtered_emotions.values()), marker='o')
    # plt.title("Emotion Changes Over Time")
    # plt.xlabel("Time")
    # plt.ylabel("Frequency")
    # plt.xticks(rotation=45)
    # plt.ylim(0, max(filtered_emotions.values(), default=1) + 1)

    # bar chat (for frequent emotion)
    plt.subplot(1, 2, 2)
    plt.bar(filtered_emotions.keys(), filtered_emotions.values(), color='skyblue')
    plt.title("Most Detected Emotions")
    plt.xticks(rotation=45)
    plt.ylabel("Count")

    plt.tight_layout()
    plt.draw()


# real time webcam emotion and face detection
def real_time_emotion_detection(model_path):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model_path, device)

    capture = cv2.VideoCapture(0) # select the default webcam

    # live graph
    figure2 = plt.figure(figsize=(10, 5))
    anim = animation.FuncAnimation(figure2, update_graphs, interval=1000, cache_frame_data=False)
    #threading.Thread(target=plt.show, daemon=True).start()
    plt.show(block=False)  # updating the graph without blocking the main thread


    while True:
        ret, frame = capture.read()
        if not ret:
            break # no frame capture, exit loop
        
        # convert frame to grayscale (haar - grayscale images)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 

        # DETECT FACE
        faces = FACE_CASCADE.detectMultiScale(gray, scaleFactor= 1.3, minNeighbors= 5, minSize= (50, 50))


        for (x, y, w, h) in faces:
            face_roi = frame [y: y+h, x: x+w] # to crop the face area

            # predict emotion
            # prediction, dominant_emotion= predict_emotion(face_roi, model, device)    
            prediction, emotions, confidences, dominant_emotion= predict_emotion(face_roi, model, device)    

            # update garaph emotions
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            time_window.append(timestamp)
            emotion_counts[dominant_emotion] += 1

            # log it to csv file
            log_emotion(timestamp, emotions, confidences)

            # draw a box 
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # display predicted emotion text
            cv2.putText(frame, prediction, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # display webcam feed
        cv2.imshow("Real time Emotion Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"): # q for quit
            break

        plt.pause(0.01) # force matplotlib to refresh/update the graph

       

        

    # remove all resources
    capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    real_time_emotion_detection("./models/emotion_model.pth")

# Change model loading path
# model = torch.load("../MODELS/emotion_model.pth")    