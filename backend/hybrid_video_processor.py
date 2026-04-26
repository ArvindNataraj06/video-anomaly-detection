import cv2
import os
import time
import torch
import numpy as np
from datetime import datetime
from collections import Counter
from ultralytics import YOLO

from model import VideoAutoencoder


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def calculate_final_severity(anomaly_percentage, max_error):
    if anomaly_percentage == 0:
        return "NORMAL"
    elif anomaly_percentage < 10 and max_error < 0.08:
        return "LOW"
    elif anomaly_percentage < 30 and max_error < 0.12:
        return "MEDIUM"
    return "HIGH"


def analyze_video_hybrid(video_path, output_dir="outputs"):
    os.makedirs(output_dir, exist_ok=True)

    start_time = time.time()

    # Load models
    autoencoder = VideoAutoencoder().to(device)
    autoencoder.load_state_dict(torch.load("models/autoencoder.pth", map_location=device))
    autoencoder.eval()

    yolo_model = YOLO("yolov8s.pt")  # lightweight pretrained YOLO model

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError("Could not open video file")

    total_frames = 0
    processed_frames = 0
    anomaly_frames = []
    all_errors = []
    all_latencies = []
    detected_objects_counter = Counter()
    severity_counter = Counter()

    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            total_frames += 1

            # Process every 5th frame for speed
            if total_frames % 5 != 0:
                continue

            frame_start = time.time()
            processed_frames += 1

            # -------------------------
            # 1. Autoencoder anomaly score
            # -------------------------
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (64, 64))
            normalized = resized / 255.0

            input_tensor = (
                torch.tensor(normalized)
                .unsqueeze(0)
                .unsqueeze(0)
                .float()
                .to(device)
            )

            output = autoencoder(input_tensor)
            reconstruction_error = torch.mean((output - input_tensor) ** 2).item()
            all_errors.append(reconstruction_error)

            # -------------------------
            # 2. YOLO object detection
            # -------------------------
            yolo_results = yolo_model(frame, verbose=False)

            objects_in_frame = []

            for result in yolo_results:
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    label = yolo_model.names[class_id]

                    if confidence < 0.25:  # filter out low confidence detections
                        continue

                    objects_in_frame.append(label)
                    detected_objects_counter[label] += 1

                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        frame,
                        f"{label} {confidence:.2f}",
                        (x1, max(y1 - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2,
                    )

            # -------------------------
            # 3. Hybrid anomaly decision
            # -------------------------
            is_anomaly = reconstruction_error >= 0.080
            frame_severity = "HIGH" if reconstruction_error >= 0.120 else "MEDIUM" if is_anomaly else "NORMAL"
            severity_counter[frame_severity] += 1

            if is_anomaly and total_frames % 20 == 0:
                object_text = ", ".join(sorted(set(objects_in_frame))) if objects_in_frame else "no major object detected"

                reason = f"High reconstruction error with {object_text}"

                filename = (
                    f"hybrid_frame_{total_frames}_"
                    f"{frame_severity}_"
                    f"{datetime.now().strftime('%H%M%S')}.jpg"
                )

                output_path = os.path.join(output_dir, filename)

                cv2.putText(frame, f"Hybrid Anomaly | {frame_severity} | Error: {reconstruction_error:.4f}",
                (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                cv2.putText(frame, reason[:70],
                (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                cv2.imwrite(output_path, frame)

                anomaly_frames.append({
                    "frame_number": total_frames,
                    "severity": frame_severity,
                    "reconstruction_error": round(reconstruction_error, 6),
                    "objects": sorted(set(objects_in_frame)),
                    "reason": reason,
                    "saved_frame": output_path
                })

            latency_ms = (time.time() - frame_start) * 1000
            all_latencies.append(latency_ms)

    cap.release()

    total_time = time.time() - start_time
    anomaly_percentage = (len(anomaly_frames) / processed_frames) * 100 if processed_frames > 0 else 0
    avg_error = float(np.mean(all_errors)) if all_errors else 0.0
    max_error = float(np.max(all_errors)) if all_errors else 0.0
    avg_latency = float(np.mean(all_latencies)) if all_latencies else 0.0
    fps = processed_frames / total_time if total_time > 0 else 0.0

    final_severity = calculate_final_severity(anomaly_percentage, max_error)

    return {
        "method": "YOLO + Autoencoder Hybrid Detection",
        "total_frames": total_frames,
        "processed_frames": processed_frames,
        "frame_skip": 5,
        "objects_detected": dict(detected_objects_counter),
        "anomaly_frames_count": len(anomaly_frames),
        "anomaly_percentage": round(anomaly_percentage, 2),
        "average_reconstruction_error": round(avg_error, 6),
        "max_reconstruction_error": round(max_error, 6),
        "average_latency_ms": round(avg_latency, 2),
        "fps": round(fps, 2),
        "severity_distribution": dict(severity_counter),
        "final_severity": final_severity,
        "evidence_frames": anomaly_frames[:10],
    }