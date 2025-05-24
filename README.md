# SignLanguageTranslator
# Sign Language Translator using Mediapipe and LSTM

This project is a real-time sign language translator built using Mediapipe for hand landmark detection and a TensorFlow LSTM model for gesture recognition. The system captures hand gestures via webcam, identifies the sign being shown, and converts it into text and speech in real time.

## Features

- Real-time hand landmark detection using Mediapipe
- LSTM-based gesture recognition from sequences of landmarks
- Smooth and debounced predictions using a moving window
- Converts recognized signs to speech using text-to-speech
- Sentence construction with options to reset, speak, and save
- Modular, threaded implementation for responsive performance

## Technologies Used

- Python
- Mediapipe
- TensorFlow / Keras
- OpenCV
- pyttsx3 (for TTS)
- NumPy
- Multithreading

## How It Works

1. Captures video feed from webcam.
2. Uses Mediapipe to detect hand landmarks.
3. Collects a sequence of 20 landmark frames.
4. Feeds this sequence to an LSTM model for prediction.
5. Displays the recognized sign, updates sentence history, and optionally speaks or saves it.

## Installation

```bash
git clone https://github.com/yourusername/SignLanguageTranslator.git
cd SignLanguageTranslator
pip install -r requirements.txt
