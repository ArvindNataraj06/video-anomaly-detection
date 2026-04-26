# 🚦 Traffic Video Intelligence System

An end-to-end AI-powered traffic video analytics system that processes CCTV or traffic videos to detect objects, track movement, estimate traffic patterns, and analyze traffic risk.

Built using **FastAPI, OpenCV, YOLOv8, PyTorch, and React**, this system simulates a real-world traffic monitoring solution.

---

## 📌 Overview

This project allows users to upload traffic videos through a frontend dashboard and receive detailed analytics including:

* Traffic object detection
* Unique object tracking
* Movement direction estimation
* Traffic density calculation
* Risk level classification
* Evidence frame generation

---

## ✨ Features

* 🎥 Upload traffic video via React frontend
* 🚗 Detect vehicles and pedestrians using YOLOv8
* 🔁 Track unique objects using YOLO tracking IDs
* 🧭 Estimate movement directions:

  * Left to right
  * Right to left
  * Upward / Downward
  * Stationary
* 📊 Calculate:

  * Total object detections across frames
  * Unique tracked objects
  * Average objects per frame
  * Maximum objects in a frame
* ⚠️ Traffic risk analysis based on:

  * Density
  * Stationary percentage
  * Pedestrian-vehicle interaction
* 🖼️ Save and display evidence frames
* ⚡ Display FPS and performance metrics
* 🌐 Full-stack integration with React UI

---

## 🧱 Tech Stack

### 🔹 Backend

* Python
* FastAPI
* OpenCV
* YOLOv8 (Ultralytics)
* PyTorch
* NumPy

### 🔹 Frontend

* React (Vite)
* JavaScript
* CSS

---

## 🏗️ Architecture

```text
Video Upload (Frontend)
        ↓
FastAPI Backend
        ↓
Frame Extraction (OpenCV)
        ↓
YOLOv8 Object Detection
        ↓
YOLO Tracking (Unique IDs)
        ↓
Direction Estimation (Center Tracking)
        ↓
Traffic Risk Analysis (Rule-Based)
        ↓
Evidence Frame Saving
        ↓
React Dashboard Display
```

---

## 🔌 API Endpoint

```http
POST /analyze-traffic-video
```

---

## 📊 Example Output

```json
{
  "method": "Traffic Object Detection, Counting, Direction Estimation and Risk Analysis",
  "total_frames": 5108,
  "processed_frames": 1021,
  "frame_skip": 5,
  "traffic_object_detections_across_frames": {
    "person": 2594,
    "motorcycle": 1063,
    "truck": 1027
  },
  "unique_tracked_traffic_objects": {
    "person": 213,
    "motorcycle": 113,
    "truck": 79
  },
  "movement_directions": {
    "left_to_right": 564,
    "stationary": 3534,
    "right_to_left": 317
  },
  "traffic_risk_analysis": {
    "traffic_density_level": "MEDIUM",
    "stationary_percentage": 78.31,
    "dominant_direction": "left_to_right",
    "final_traffic_risk": "HIGH"
  }
}
```

---

## ⚙️ Setup & Run

### 🔹 Backend

```bash
cd backend
pip install -r requirements.txt
python -m uvicorn main:app --reload
```

Backend runs at:

```text
http://127.0.0.1:8000
```

---

### 🔹 Frontend

```bash
cd frontend
npm install
npm run dev
```

Frontend runs at:

```text
http://localhost:5173
```

---

## 📁 Project Structure

```text
video-anomaly-detection/
│
├── backend/
│   ├── main.py
│   ├── traffic_video_processor.py
│   ├── outputs/
│
├── frontend/
│   ├── src/
│   ├── App.jsx
│   ├── App.css
│
├── README.md
```

---

## ⚠️ Limitations

* Detection accuracy depends on video quality and camera angle
* Tracking is based on YOLO IDs and may vary in crowded scenes
* Direction estimation is rule-based (not deep tracking)
* Traffic risk analysis is heuristic (not trained ML model)

---

## 🚀 Future Improvements

* Wrong-way vehicle detection
* Lane-based traffic analysis
* License plate recognition
* Real-time CCTV / RTSP stream support
* Advanced tracking (DeepSORT / ByteTrack)
* Model evaluation using labeled datasets
* Deployment using Docker + Cloud

---

## 📌 Author

**Arvind Nataraj**
MSc Applied Computer Science
Software Developer & AI Engineer

---
