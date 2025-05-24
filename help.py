import cv2
import mediapipe as mp
import numpy as np
import os

# Mediapipe hands initialization
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.7,
                       min_tracking_confidence=0.5)

# Directory with your sign videos or frames organized by label folders
DATA_DIR =r"C:\Users\Dell\Desktop\SIGN PROJECT\lsa64_cut"
OUTPUT_DIR = r"C:\Users\Dell\Desktop\SIGN PROJECT\landmark_sequences"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def extract_landmarks_from_frame(frame):
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]  # Use first detected hand
        # Extract 21 landmarks (x,y,z) normalized coordinates
        landmarks = []
        for lm in hand_landmarks.landmark:
            landmarks.extend([lm.x, lm.y, lm.z])
        return landmarks
    else:
        return None

def process_video(video_path, save_path):
    cap = cv2.VideoCapture(video_path)
    sequence = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        landmarks = extract_landmarks_from_frame(frame)
        if landmarks is not None:
            sequence.append(landmarks)

    cap.release()

    if len(sequence) > 0:
        sequence = np.array(sequence)
        np.save(save_path, sequence)  # Save numpy array of shape (num_frames, 63)
        print(f"Saved landmark sequence to {save_path}")
    else:
        print(f"No landmarks found in {video_path}")

# Example: processing all videos in folders      each sign label
for label in os.listdir(DATA_DIR):
    label_dir = os.path.join(DATA_DIR, label)
    if os.path.isdir(label_dir):
        save_label_dir = os.path.join(OUTPUT_DIR, label)
        os.makedirs(save_label_dir, exist_ok=True)

        for video_file in os.listdir(label_dir):
            if video_file.endswith(('.mp4', '.avi', '.mov')):
                video_path = os.path.join(label_dir, video_file)
                save_path = os.path.join(save_label_dir, os.path.splitext(video_file)[0] + '.npy')
                process_video(video_path, save_path)

hands.close()
