import cv2
import os
import torch
import numpy as np
from datetime import datetime
from model import VideoAutoencoder


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_ai_severity(error):
    if error < 0.055:
        return "NORMAL"
    elif error < 0.065:
        return "LOW"
    elif error < 0.080:
        return "MEDIUM"
    return "HIGH"


def analyze_video_ai(video_path, output_dir="outputs"):
    os.makedirs(output_dir, exist_ok=True)

    model = VideoAutoencoder().to(device)
    model.load_state_dict(torch.load("models/autoencoder.pth", map_location=device))
    model.eval()

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError("Could not open video file")

    frame_count = 0
    anomaly_frames = []
    anomaly_scores = []
    all_errors = []

    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

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

            output = model(input_tensor)
            error = torch.mean((output - input_tensor) ** 2).item()
            all_errors.append(error)

            severity = get_ai_severity(error)

            if severity in ["MEDIUM", "HIGH"]:
                filename = f"ai_anomaly_frame_{frame_count}_{datetime.now().strftime('%H%M%S')}.jpg"
                output_path = os.path.join(output_dir, filename)

                cv2.putText(
                    frame,
                    f"AI Anomaly | Severity: {severity} | Error: {error:.4f}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 0, 255),
                    2,
                )

                cv2.imwrite(output_path, frame)
                anomaly_frames.append(output_path)
                anomaly_scores.append(error)

        cap.release()

    avg_error = float(np.mean(all_errors)) if all_errors else 0.0
    max_error = float(np.max(all_errors)) if all_errors else 0.0

    anomaly_percentage = (len(anomaly_frames) / frame_count) * 100 if frame_count > 0 else 0

    if anomaly_percentage == 0:
        final_severity = "NORMAL"
    elif anomaly_percentage < 5:
        final_severity = "LOW"
    elif anomaly_percentage < 25:
        final_severity = "MEDIUM"
    else:
        final_severity = "HIGH"

    return {
        "method": "Autoencoder Reconstruction Error",
        "total_frames_processed": frame_count,
        "total_ai_anomalies_detected": len(anomaly_frames),
        "anomaly_percentage": round(anomaly_percentage, 2),
        "average_reconstruction_error": round(avg_error, 6),
        "max_reconstruction_error": round(max_error, 6),
        "final_severity": final_severity,
        "anomaly_frames": anomaly_frames[:10],
    }