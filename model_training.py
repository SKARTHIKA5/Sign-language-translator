import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pickle
import matplotlib.pyplot as plt

# Load Gesture Data
data_path = "C:/Users/dell/Videos/sign_language/balanced_gesture_data.csv"  # Update with your correct path
data = pd.read_csv(data_path)

# Extract Features (Hand Landmarks) & Labels
y = data["gesture"].values
X = data.drop(columns=["gesture"]).values  # Remove the label column

# Normalize Features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Encode Labels to Numbers
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Convert y_train and y_test to categorical (one-hot encoding)
num_classes = len(np.unique(y))  # Get the number of unique gestures
y = to_categorical(y, num_classes=num_classes)

# Split Data into Training & Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Updated y_train shape:", y_train.shape)

# Reshape Data for LSTM (Adding Time Step Dimension)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Build LSTM Model with Dropout
model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.3),
    LSTM(64),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(num_classes, activation='softmax')  # Ensure it matches the number of classes
])

# Compile Model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Callbacks: Early Stopping & ReduceLROnPlateau
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)

# Train Model
history = model.fit(X_train, y_train, validation_data=(X_test, y_test),
                    epochs=50, batch_size=32, 
                    callbacks=[early_stop, lr_scheduler])

# Evaluate Model
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc * 100:.2f}%")

# Save Model & Encoders
model.save("C:/Users/dell/Videos/sign_language/gesture_recognition_model.h5")
pickle.dump(scaler, open("C:/Users/dell/Videos/sign_language/scaler.pkl", "wb"))
pickle.dump(label_encoder, open("C:/Users/dell/Videos/sign_language/label_encoder.pkl", "wb"))

print("Model training complete! Model saved for real-time testing.")

# Plot Training vs Validation Accuracy
plt.figure(figsize=(8, 5))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training vs Validation Accuracy')
plt.show()

# Plot Training vs Validation Loss
plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training vs Validation Loss')
plt.show()
