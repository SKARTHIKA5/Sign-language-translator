import os
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import pickle

# Load trained model & encoders
model = tf.keras.models.load_model("gesture_recognition_model.h5")
scaler = pickle.load(open("C:/Users/dell/Videos/sign_language/scaler.pkl", "rb"))
label_encoder = pickle.load(open("C:/Users/dell/Videos/sign_language/label_encoder.pkl", "rb"))

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# Start Video Capture
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame (mirror effect)
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process hand landmarks
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

            # Ensure consistent input shape
            landmarks = np.array(landmarks).reshape(1, -1)
            landmarks = scaler.transform(landmarks)  # Normalize
            landmarks = np.expand_dims(landmarks, axis=2)  # Reshape for LSTM model

            # Make a prediction
            prediction = model.predict(landmarks)
            confidence = np.max(prediction)

            if confidence > 0.7:  # Only show high-confidence predictions
                class_id = np.argmax(prediction)
                gesture = label_encoder.inverse_transform([class_id])[0]
            else:
                gesture = "Uncertain"

            # Display gesture on screen
            cv2.putText(frame, f"Gesture: {gesture} ({confidence:.2f})", (10, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            # Draw hand landmarks
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Show the video frame
    cv2.imshow("Real-Time Gesture Recognition", frame)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
