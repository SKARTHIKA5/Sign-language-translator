import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf

import cv2
import mediapipe as mp
import pandas as pd
import os

# Initialize MediaPipe Hands module
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils  # For drawing hand landmarks
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Open the webcam
cap = cv2.VideoCapture(0)  # Change to 1 if using an external camera

# CSV file to store hand landmarks for AI training
CSV_FILE = "C:\\Users\\dell\\Videos\\sign_language\\gesture_data.csv"


# Set a label for the current gesture (change this for each gesture you collect)
GESTURE_LABEL = "X"  # Change this label when collecting different gestures

while cap.isOpened():
    ret, frame = cap.read()  # Read a frame from the webcam
    frame = cv2.flip(frame, 1)  # Fp for a mirror effect
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB (needed for MediaPipe)
    
    # Process the frame for hand detection
    results = hands.process(rgb_frame)

    # Prepare data storage
    hand_data = []

    # If hands are detected, extract key points
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw landmarks on the screen
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Extract landmark coordinates
            for lm in hand_landmarks.landmark:
                hand_data.extend([lm.x, lm.y, lm.z])  # Store all (x, y, z) values in a list
            
            # Save hand data if 21 landmarks are detected
            if len(hand_data) == 63:  # 21 landmarks * (x, y, z)
                df = pd.DataFrame([hand_data])
                df['gesture'] = GESTURE_LABEL  # Label the gesture
                if not os.path.exists(CSV_FILE):
                    df.to_csv(CSV_FILE, mode='w', header=True, index=False)  # Create new file with headers
                else:
                    df.to_csv(CSV_FILE, mode='a', header=False, index=False)  # Append without headers
                # Append to CSV file

    # Display the frame
    cv2.imshow("Hand Tracking", frame)
    
    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
