
import os
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import pickle
import pyttsx3
from sklearn.preprocessing import StandardScaler

# Disable OneDNN optimizations for TensorFlow
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Initialize Text-to-Speech engine
tts_engine = pyttsx3.init()
tts_engine.setProperty('rate', 150)  # Speech speed
tts_engine.setProperty('volume', 1.0)  # Volume level

def speak_text(text):
    """Convert recognized gesture text to speech."""
    tts_engine.say(text)
    tts_engine.runAndWait()

# Load trained model & preprocessing tools
model = tf.keras.models.load_model("gesture_recognition_model.h5")
scaler = pickle.load(open("scaler.pkl", "rb"))
label_encoder = pickle.load(open("label_encoder.pkl", "rb"))

# Initialize MediaPipe Hands for gesture tracking
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

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

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Extract hand landmarks (x, y, z coordinates)
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

            # Convert landmarks to numpy array & normalize
            landmarks = np.array(landmarks).reshape(1, -1)
            landmarks = scaler.transform(landmarks)  # Scale the input
            landmarks = np.expand_dims(landmarks, axis=2)  # Reshape for model

            # Make a prediction
            prediction = model.predict(landmarks)
            class_id = np.argmax(prediction)
            gesture = label_encoder.inverse_transform([class_id])[0]

            # Display recognized gesture on screen
            cv2.putText(frame, f"Gesture: {gesture}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                        1, (0, 255, 0), 2, cv2.LINE_AA)

            # Speak the recognized text
            speak_text(gesture)

            # Draw hand landmarks
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Show video frame
    cv2.imshow("Real-Time Gesture Recognition", frame)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

