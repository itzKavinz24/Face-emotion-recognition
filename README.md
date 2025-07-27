# 😊 Face Emotion Recognition using OpenCV

This project uses **OpenCV** and a pre-trained **deep learning model** to detect human faces in real-time via webcam and classify their **emotions** (e.g., Happy, Sad, Angry, Surprise, etc.).

## 🔍 Features

- Real-time face detection using Haar cascades
- Emotion classification using CNN-based model (e.g., FER2013-trained model)
- Easy to use with webcam input
- Clean and user-friendly UI window with emotion label overlay

## 📂 Project Structure

face-emotion-recognition/
│
├── haarcascade_frontalface_default.xml # Face detection model
├── emotion_model.h5 # Pre-trained emotion recognition model
├── recognize_emotion.py # Main Python script
├── requirements.txt # List of required Python packages
└── README.md # Project info


## 🛠️ Requirements

- Python 3.7+
- OpenCV (`opencv-python`)
- TensorFlow / Keras
- NumPy

Install all dependencies using:

```bash
pip install -r requirements.txt
