import os
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import pickle
import pyttsx3
import time
from collections import deque

# Disable OneDNN optimizations
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Initialize Text-to-Speech engine
tts_engine = pyttsx3.init()
tts_engine.setProperty('rate', 150)
tts_engine.setProperty('volume', 1.0)

# Load trained model & preprocessing tools
model = tf.keras.models.load_model("gesture_recognition_model.h5")
scaler = pickle.load(open("scaler.pkl", "rb"))
label_encoder = pickle.load(open("label_encoder.pkl", "rb"))

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# Buffers to store recognized letters & words
letter_buffer = deque(maxlen=15)
word_buffer = []
sentence_buffer = []

# Timing for word separation
last_gesture_time = time.time()

# Function to convert text to speech
def speak_text(text):
    tts_engine.say(text)
    tts_engine.runAndWait()

# Start Video Capture
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip frame (mirror effect)
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process hand landmarks
    results = hands.process(rgb_frame)

    current_time = time.time()

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Extract hand landmarks
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

            # Normalize input
            landmarks = np.array(landmarks).reshape(1, -1)
            landmarks = scaler.transform(landmarks)
            landmarks = np.expand_dims(landmarks, axis=2)

            # Make a prediction
            prediction = model.predict(landmarks)
            class_id = np.argmax(prediction)
            gesture = label_encoder.inverse_transform([class_id])[0]

            # If a new gesture is detected, update buffer and reset timer
            if gesture not in letter_buffer:
                letter_buffer.append(gesture)
                last_gesture_time = current_time

            # Display recognized gesture
            cv2.putText(frame, f"Letter: {gesture}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                        1, (0, 255, 0), 2, cv2.LINE_AA)

            # Draw hand landmarks
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # If no gesture detected for 1.5 seconds, assume a word break
    if (current_time - last_gesture_time) > 1.5 and letter_buffer:
        word = "".join(letter_buffer)
        word_buffer.append(word)
        letter_buffer.clear()
        print(f"Word: {word}")

    # Show video frame
    cv2.imshow("ASL Word & Sentence Formation", frame)

    # Handle keyboard input
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):  # Exit on 'q'
        break
    elif key == ord(' '):  # Spacebar adds a space
        word_buffer.append("")
    elif key == 13:  # Enter finalizes sentence
        sentence = " ".join(word_buffer)
        sentence_buffer.append(sentence)
        print(f"Sentence: {sentence}")
        speak_text(sentence)
        word_buffer.clear()
    elif key == 8:  # Backspace removes last letter or word
        if letter_buffer:
            letter_buffer.pop()
        elif word_buffer:
            word_buffer.pop()

# Release resources
cap.release()
cv2.destroyAllWindows()
