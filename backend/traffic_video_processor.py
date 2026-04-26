import cv2
import os
import time
import math
from collections import Counter
from ultralytics import YOLO


yolo_model = YOLO("yolov8s.pt")

TRAFFIC_CLASSES = ["car", "truck", "bus", "motorcycle", "bicycle", "person"]


def get_center(box):
    x1, y1, x2, y2 = map(int, box)
    return ((x1 + x2) // 2, (y1 + y2) // 2)


def estimate_direction(old_center, new_center, movement_threshold=15):
    old_x, old_y = old_center
    new_x, new_y = new_center

    dx = new_x - old_x
    dy = new_y - old_y

    if abs(dx) < movement_threshold and abs(dy) < movement_threshold:
        return "stationary"

    if abs(dx) > abs(dy):
        return "left_to_right" if dx > 0 else "right_to_left"
    else:
        return "downward" if dy > 0 else "upward"


def euclidean_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

def calculate_traffic_risk(avg_objects, max_objects, direction_counts, unique_object_counts):
    risk_reasons = []

    total_directions = sum(direction_counts.values())
    stationary_count = direction_counts.get("stationary", 0)
    stationary_percentage = (stationary_count / total_directions) * 100 if total_directions > 0 else 0

    moving_directions = {
        k: v for k, v in direction_counts.items()
        if k != "stationary"
    }

    dominant_direction = (
        max(moving_directions, key=moving_directions.get)
        if moving_directions else "not_enough_movement"
    )

    if avg_objects < 4:
        traffic_density_level = "LOW"
    elif avg_objects < 8:
        traffic_density_level = "MEDIUM"
        risk_reasons.append("Moderate traffic density detected")
    else:
        traffic_density_level = "HIGH"
        risk_reasons.append("High traffic density detected")

    if max_objects >= 12:
        risk_reasons.append("High object concentration detected in at least one frame")

    if stationary_percentage > 65:
        risk_reasons.append("High stationary object percentage may indicate slow traffic or congestion")

    person_count = unique_object_counts.get("person", 0)
    vehicle_count = (
        unique_object_counts.get("car", 0)
        + unique_object_counts.get("truck", 0)
        + unique_object_counts.get("bus", 0)
        + unique_object_counts.get("motorcycle", 0)
        + unique_object_counts.get("bicycle", 0)
    )

    if person_count > 20 and vehicle_count > 20:
        risk_reasons.append("Pedestrians and vehicles detected together, possible interaction risk")

    if traffic_density_level == "HIGH" or stationary_percentage > 75:
        final_traffic_risk = "HIGH"
    elif traffic_density_level == "MEDIUM" or stationary_percentage > 45 or len(risk_reasons) >= 2:
        final_traffic_risk = "MEDIUM"
    else:
        final_traffic_risk = "LOW"

    if not risk_reasons:
        risk_reasons.append("Traffic flow appears normal based on current rules")

    return {
        "traffic_density_level": traffic_density_level,
        "stationary_percentage": round(stationary_percentage, 2),
        "dominant_direction": dominant_direction,
        "final_traffic_risk": final_traffic_risk,
        "risk_reasons": risk_reasons
    }

def analyze_traffic_video(video_path, output_dir="outputs"):
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError("Could not open video file")

    total_frames = 0
    processed_frames = 0

    
    traffic_object_counts = Counter()
    frame_object_counts = []
    direction_counts = Counter()
    unique_objects = set()
    unique_object_counts = Counter()
    evidence_frames = []

    previous_objects = []
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        total_frames += 1

        if total_frames % 5 != 0:
            continue

        processed_frames += 1

        results = yolo_model.track(frame, persist=True, verbose=False)

        current_frame_count = 0
        objects_in_frame = []
        current_objects = []

        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                label = yolo_model.names[class_id]

                if label not in TRAFFIC_CLASSES:
                    continue

                if confidence < 0.25:
                    continue

                track_id = None
                if box.id is not None:
                    track_id = int(box.id[0])
                    unique_key = f"{label}_{track_id}"

                if unique_key not in unique_objects:
                    unique_objects.add(unique_key)
                    unique_object_counts[label] += 1

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                center = get_center((x1, y1, x2, y2))

                traffic_object_counts[label] += 1
                current_frame_count += 1
                objects_in_frame.append(label)

                direction = "new_object"

                best_match = None
                best_distance = float("inf")

                for prev_obj in previous_objects:
                    if prev_obj["label"] != label:
                        continue

                    distance = euclidean_distance(center, prev_obj["center"])

                    if distance < best_distance:
                        best_distance = distance
                        best_match = prev_obj

                if best_match and best_distance < 80:
                    direction = estimate_direction(best_match["center"], center)
                    direction_counts[direction] += 1

                current_objects.append({
                    "label": label,
                    "center": center
                })

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                cv2.circle(frame, center, 4, (255, 0, 0), -1)

                cv2.putText(
                    frame,
                    f"{label} {confidence:.2f} {direction}",
                    (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 0),
                    2,
                )

        frame_object_counts.append(current_frame_count)
        previous_objects = current_objects

        if processed_frames % 20 == 0:
            filename = f"traffic_direction_frame_{processed_frames}.jpg"
            output_path = os.path.join(output_dir, filename)

            cv2.putText(
                frame,
                f"Objects: {current_frame_count}",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )

            cv2.imwrite(output_path, frame)

            evidence_frames.append({
                "frame_number": total_frames,
                "object_count": current_frame_count,
                "objects": sorted(set(objects_in_frame)),
                "saved_frame": output_path.replace("\\", "/")
            })

    cap.release()

    total_time = time.time() - start_time

    avg_objects = sum(frame_object_counts) / len(frame_object_counts) if frame_object_counts else 0
    max_objects = max(frame_object_counts) if frame_object_counts else 0
    fps = processed_frames / total_time if total_time > 0 else 0

    traffic_risk = calculate_traffic_risk(
    avg_objects,
    max_objects,
    direction_counts,
    unique_object_counts
)

    return {
    "method": "Traffic Object Detection, Counting, Direction Estimation and Risk Analysis",
    "total_frames": total_frames,
    "processed_frames": processed_frames,
    "frame_skip": 5,
    "traffic_object_detections_across_frames": dict(traffic_object_counts),
    "unique_tracked_traffic_objects": dict(unique_object_counts),
    "movement_directions": dict(direction_counts),
    "average_objects_per_frame": round(avg_objects, 2),
    "max_objects_in_frame": max_objects,
    "fps": round(fps, 2),
    "traffic_risk_analysis": traffic_risk,
    "evidence_frames": evidence_frames[:10]
}