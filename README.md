# AI- Powered sign language translator: A real-time vision and NLP based Communicator
A computer vision-powered AI project that uses hand gestures captured from a webcam and translates them into spoken and written language in real time. This system is designed to assist individuals with speech or hearing impairments, and to create inclusive, gesture-based human-computer interaction.

# Overview
This project recognizes static hand gestures (like ASL alphabets or predefined words) using a live video feed, and intelligently forms words and sentences. It then converts the recognized text into speech using offline text-to-speech (TTS).

The complete pipeline includes:

Gesture data collection

Dataset balancing

LSTM model training

Real-time prediction using MediaPipe & OpenCV

Word/sentence buffering logic

Speech generation
