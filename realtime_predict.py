import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import pyttsx3
from threading import Thread, Lock
from queue import Queue
import time
import logging
from collections import deque, Counter

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class SignLanguageTranslator:
    def __init__(self):
        self.SEQUENCE_LENGTH = 20
        self.label_map = {
            i + 1: label for i, label in enumerate([
                "Thank you", "Hello", "Goodbye", "Please", "Yes", "No", "How?", "What?", "Where?", "Who?",
                "Why?", "Water", "Food", "Family", "Listen", "Read", "Walk", "Sing", "Work", "Play",
                "Love", "Like", "Feel", "Look", "Speak", "Understand", "Help", "Drink", "Buy", "Sell",
                "Study", "Cold", "Fast", "Baby", "Good", "Bad", "Happy", "Sad", "Big", "Small",
                "House", "School", "Friend", "Book", "Music", "Sleep", "Run", "Dance", "Learn", "Teach",
                "Ask", "Answer", "Cook", "Travel", "Wait", "Eat", "Home", "Need"
            ])
        }
        self._init_components()
        self._setup_threads()
        self.sentence_lock = Lock()
        self.last_sign = None

    def _init_components(self):
        self.model = load_model("sign_lstm_model.h5")
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(max_num_hands=1,
                                       min_detection_confidence=0.7,
                                       min_tracking_confidence=0.7)
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 160)
        self.engine.setProperty('volume', 0.9)
        if len(self.engine.getProperty('voices')) > 1:
            self.engine.setProperty('voice', self.engine.getProperty('voices')[1].id)
        self.frame_queue = Queue(maxsize=5)
        self.output_queue = Queue(maxsize=5)
        self.speech_queue = Queue()
        self.landmark_seq = []
        self.sentence = []
        self.prediction_window = deque(maxlen=7)
        self.current_sign = ""

    def _setup_threads(self):
        self.capture_thread = Thread(target=self._capture_frames, daemon=True)
        self.processing_thread = Thread(target=self._process_frames, daemon=True)
        self.speech_thread = Thread(target=self._speak_worker, daemon=True)

    def _capture_frames(self):
        cap = cv2.VideoCapture(0)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            if not self.frame_queue.full():
                self.frame_queue.put(frame)
        cap.release()

    def _process_frames(self):
        while True:
            if self.frame_queue.empty():
                time.sleep(0.01)
                continue
            frame = self.frame_queue.get()
            self.output_queue.put(self._process_frame(frame))

    def _process_frame(self, frame):
        rgb = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)
        if results.multi_hand_landmarks:
            landmarks = [coord for lm in results.multi_hand_landmarks[0].landmark for coord in (lm.x, lm.y, lm.z)]
            preprocessed = self._preprocess_landmarks(landmarks)
            if preprocessed:
                self.landmark_seq.append(preprocessed)
                if len(self.landmark_seq) > self.SEQUENCE_LENGTH:
                    self.landmark_seq.pop(0)
            if len(self.landmark_seq) == self.SEQUENCE_LENGTH:
                frame = self._predict_sign(frame)
            else:
                frame = self._draw_text(frame, "Collecting data...", (20, 50), (0, 255, 255))
        else:
            self.landmark_seq.clear()
            self.current_sign = ""
        return self._draw_sentence_and_ui(frame)

    def _preprocess_landmarks(self, landmarks):
        try:
            landmarks = np.array(landmarks).reshape(21, 3)
            landmarks -= landmarks[0]
            return landmarks.flatten().tolist()
        except Exception as e:
            logging.warning(f"Landmark error: {e}")
            return []

    def _predict_sign(self, frame):
        input_data = np.expand_dims(np.array(self.landmark_seq), axis=0)
        prediction = self.model.predict(input_data, verbose=0)[0]
        pred_idx = np.argmax(prediction)
        confidence = prediction[pred_idx]
        self.prediction_window.append(pred_idx)
        smoothed_idx = Counter(self.prediction_window).most_common(1)[0][0]

        # Try both lines below if you have issues with "Unknown"
        self.current_sign = self.label_map.get(smoothed_idx + 1, "Unknown")
        # self.current_sign = self.label_map.get(smoothed_idx, "Unknown")

        print(f"Predicted index: {pred_idx}, Smoothed index: {smoothed_idx}, Sign: {self.current_sign}, Confidence: {confidence:.2f}")

        with self.sentence_lock:
            if self.current_sign != "Unknown" and self.current_sign != self.last_sign:
                self.sentence.append(self.current_sign)
                self.last_sign = self.current_sign
                print(f"ADDED: {self.current_sign}")

        color = (0, 255, 0) if confidence > 0.7 else (0, 165, 255)
        return self._draw_text(frame, f"{self.current_sign} ({confidence:.2f})", (20, 50), color)

    def _draw_text(self, frame, text, position, color=(255, 255, 255), font_scale=1.2, thickness=2):
        cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)
        return frame

    def _draw_sentence_and_ui(self, frame):
        overlay = frame.copy()
        h, w = frame.shape[:2]
        cv2.rectangle(overlay, (0, h - 120), (w, h), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
        with self.sentence_lock:
            sentence_text = ' '.join(self.sentence[-10:])
        self._draw_text(frame, sentence_text, (10, h - 50))
        self._draw_text(frame, "SPACE: Speak | R: Reset | S: Save | ESC: Exit", (10, h - 15), (180, 180, 180), 0.7, 2)
        return frame

    def _speak_worker(self):
        while True:
            text = self.speech_queue.get()
            if text:
                self.engine.say(text)
                self.engine.runAndWait()
            self.speech_queue.task_done()

    def _speak_sentence(self):
        with self.sentence_lock:
            if self.sentence:
                text = ' '.join(self.sentence)
                self.speech_queue.put(text)
            else:
                logging.warning("Nothing to speak")

    def _save_sentence(self):
        with self.sentence_lock:
            if self.sentence:
                filename = f"signs_{time.strftime('%Y%m%d-%H%M%S')}.txt"
                with open(filename, "w") as f:
                    f.write(' '.join(self.sentence))
                logging.info(f"Saved to {filename}")
            else:
                logging.warning("No signs to save - make signs first!")

    def run(self):
        self.capture_thread.start()
        self.processing_thread.start()
        self.speech_thread.start()
        try:
            while True:
                if not self.output_queue.empty():
                    cv2.imshow("Sign Language Translator", self.output_queue.get())
                key = cv2.waitKey(1) & 0xFF
                if key == 27: break
                elif key == ord(' '): self._speak_sentence()
                elif key == ord('r'):
                    with self.sentence_lock:
                        self.sentence.clear()
                        self.last_sign = None
                    logging.info("Sentence reset")
                elif key == ord('s'): self._save_sentence()
        finally:
            self._shutdown()

    def _shutdown(self):
        self.engine.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    SignLanguageTranslator().run()
