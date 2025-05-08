


import argparse
from scripts.visual.train_resnet18 import train_model
from scripts.visual.inference_resnet18 import predict_emotion

def main():
    parser = argparse.ArgumentParser(description="Facial Emotion Detection")
    parser.add_argument('--train', action='store_true', help="Train the model")
    parser.add_argument('--inference', type=str, help="Path to an image for inference")
    args = parser.parse_args()

    if args.train:
        print("Training the model...")
        train_model()  # Call your training script here
    elif args.inference:
        print("Performing inference...")
        emotion = predict_emotion(args.inference)
        print(f"Predicted Emotion: {emotion}")
    else:
        print("Use --train to train or --inference <image_path> for inference.")

if __name__ == "__main__":
    main()
