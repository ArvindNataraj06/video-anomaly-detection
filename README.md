# Real-Time Video Anomaly Detection System

This project is an AI-powered video anomaly detection system built using Python, OpenCV, PyTorch, and FastAPI.

The system analyzes uploaded video files, detects unusual activity, calculates anomaly scores, assigns severity levels, and saves suspicious frames for review.

## Problem Statement

In places like offices, roads, malls, and public areas, it is difficult for humans to monitor CCTV footage continuously. Important abnormal events may be missed.

This project solves that problem by automatically analyzing video streams and identifying unusual patterns.

## Features

- Video upload and processing through FastAPI
- Frame-by-frame video analysis using OpenCV
- Motion-based anomaly detection baseline
- Autoencoder-based AI anomaly detection using PyTorch
- Reconstruction error-based anomaly scoring
- Severity classification: NORMAL, LOW, MEDIUM, HIGH
- Anomaly percentage calculation
- Saves detected anomaly frames as images
- Swagger API documentation

## Tech Stack

- Python
- FastAPI
- OpenCV
- PyTorch
- NumPy
- Uvicorn

## Project Structure

```text
video-anomaly-detection/
│
├── backend/
│   ├── main.py
│   ├── video_processor.py
│   ├── ai_video_processor.py
│   ├── model.py
│   ├── train_autoencoder.py
│   ├── models/
│   │   └── autoencoder.pth
│   ├── outputs/
│   └── uploads/
│
├── sample_videos/
├── README.md
└── requirements.txt
