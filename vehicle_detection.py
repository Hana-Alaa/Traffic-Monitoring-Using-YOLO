# Import dependencies
from ultralytics import YOLO
import torch
import cv2
import math
import time

# Load YOLO model (vehicle detection)
with torch.serialization.safe_globals([YOLO]):
    model = YOLO("yolov8x.pt")


# Input video
video_path = r"E:\Car Detection\videos\test1.mp4"
cap = cv2.VideoCapture(video_path)

# Video configuration
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("vehicle_count_speed.mp4", fourcc, fps, (width, height))

# Counters and tracking
count_up, count_down = 0, 0
track_last_side, prev_positions = {}, {}

# Conversion factor: pixels to meters
pixels_to_meter = 0.05
line_y = int(height * 0.55)

# Color map for vehicle types
color_map = {
    "car": (0, 255, 255),
    "truck": (255, 165, 0),
    "bus": (0, 0, 255),
    "motorcycle": (255, 0, 255),
    "bicycle": (255, 255, 0),
    "van": (0, 128, 255),
}

# Run YOLO tracking
results = model.track(
    source=video_path,
    conf=0.5,
    tracker="bytetrack.yaml",
    stream=True
)

for result in results:
    frame = result.orig_img.copy()
    cv2.line(frame, (0, line_y), (width, line_y), (0, 255, 0), 2)

    boxes = getattr(result, "boxes", None)
    if boxes is not None and len(boxes) > 0:
        xyxy = boxes.xyxy.cpu().numpy()
        ids = boxes.id.cpu().numpy() if hasattr(boxes, "id") else [None] * len(xyxy)
        classes = [model.names[int(c)] for c in boxes.cls.cpu().numpy()]

        for i, (box, tid, cls_name) in enumerate(zip(xyxy, ids, classes)):
            # Skip if no ID
            if tid is None:
                continue

            x1, y1, x2, y2 = map(int, box[:4])
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            now = time.time()

            # Speed calculation
            speed_kmh = 0
            if tid in prev_positions:
                px, py, pt = prev_positions[tid]
                dist_pixels = math.hypot(cx - px, cy - py)
                dist_meters = dist_pixels * pixels_to_meter
                dt = now - pt
                if dt > 0:
                    speed_kmh = (dist_meters / dt) * 3.6
            prev_positions[tid] = (cx, cy, now)

            # Determine direction
            curr_side = 'above' if cy < line_y else 'below'
            prev_side = track_last_side.get(tid)
            if prev_side and prev_side != curr_side:
                if prev_side == 'above' and curr_side == 'below':
                    count_down += 1
                elif prev_side == 'below' and curr_side == 'above':
                    count_up += 1
            track_last_side[tid] = curr_side

            # Draw bounding boxes and labels (ID + Type + Speed)
            color = color_map.get(cls_name.lower(), (0, 150, 255))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            label = f"{cls_name} | ID:{int(tid)} | {speed_kmh:.1f} km/h"
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    # Vehicle counters
    cv2.putText(frame, f"Down: {count_down}", (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(frame, f"Up: {count_up}", (15, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    out.write(frame)

cap.release()
out.release()
cv2.destroyAllWindows()
print("Video saved as: vehicle_count_speed.mp4")