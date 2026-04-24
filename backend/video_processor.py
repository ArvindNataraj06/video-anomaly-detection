import cv2
import os
import numpy as np
from datetime import datetime


def get_severity(score):
    if score < 20:
        return "LOW"
    elif score < 50:
        return "MEDIUM"
    return "HIGH"


def analyze_video(video_path, output_dir="outputs"):
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError("Could not open video file")

    ret, prev_frame = cap.read()

    if not ret:
        raise ValueError("Could not read first frame")

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    prev_gray = cv2.GaussianBlur(prev_gray, (21, 21), 0)

    frame_count = 0
    anomaly_frames = []
    anomaly_scores = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        frame_delta = cv2.absdiff(prev_gray, gray)
        threshold = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
        threshold = cv2.dilate(threshold, None, iterations=2)

        motion_score = np.sum(threshold) / 255 / 1000

        severity = get_severity(motion_score)

        if severity in ["MEDIUM", "HIGH"]:
            filename = f"anomaly_frame_{frame_count}_{datetime.now().strftime('%H%M%S')}.jpg"
            output_path = os.path.join(output_dir, filename)

            cv2.putText(
                frame,
                f"Anomaly Detected | Severity: {severity}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
            )

            cv2.imwrite(output_path, frame)

            anomaly_frames.append(output_path)
            anomaly_scores.append(float(motion_score))

        prev_gray = gray

    cap.release()

    total_anomalies = len(anomaly_frames)
    avg_score = float(np.mean(anomaly_scores)) if anomaly_scores else 0.0
    final_severity = get_severity(avg_score)

    return {
        "total_frames_processed": frame_count,
        "total_anomalies_detected": total_anomalies,
        "average_anomaly_score": round(avg_score, 2),
        "final_severity": final_severity if total_anomalies > 0 else "NORMAL",
        "anomaly_frames": anomaly_frames[:10],
    }