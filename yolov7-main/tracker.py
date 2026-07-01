#tracker.py
import os
import cv2
import numpy as np
import collections
import torch
import onnxruntime as ort

# Import your custom modular elements
from kpt import load_model, process_frame
from find_embeddings import keypoints_to_embedding
from utils.plots import plot_one_box

try:
    from yolox.tracker.byte_tracker import BYTETracker
except ImportError:
    import sys

    sys.path.append(r"ByteTrack-main")
    from yolox.tracker.byte_tracker import BYTETracker


class ByteTrackArgs:
    def __init__(self, fps=25):
        self.track_thresh = 0.5
        self.track_buffer = 30
        self.match_thresh = 0.8
        self.mot20 = False
        self.frame_rate = fps


# ----------------------------------------------------
# SYSTEM CONFIGURATION & TUNING
# ----------------------------------------------------
WEIGHTS_PATH = r'yolov7-w6-pose.pt'
ONNX_MODEL_PATH = r'bilstm_model.onnx'
VIDEO_PATH = r'vids/Running_demo.mp4'
OUTPUT_TXT_DIR = r'C:\Users\ayush\OneDrive\Desktop\Internship\Keypoints\Person_Embeddings'
print(VIDEO_PATH)
print(os.path.exists(VIDEO_PATH))
# --- PERFORMANCE TUNING KNOB ---
# FRAME_STEP = 1: Process every single frame natively (Slower)
# FRAME_STEP = 2: Skip every alternative frame, copying tracking states (2x Speed Up)
FRAME_STEP = 2

os.makedirs(OUTPUT_TXT_DIR, exist_ok=True)

# 1. Initialize YOLOv7-Pose Framework
device, pose_model = load_model(WEIGHTS_PATH)

# 2. Initialize BiLSTM ONNX Inference Session
print("Loading Action Classification ONNX Model...")
ort_session = ort.InferenceSession(ONNX_MODEL_PATH, providers=['CPUExecutionProvider'])
input_name = ort_session.get_inputs()[0].name

# 3. Setup Capture Stream and Track Objects
cap = cv2.VideoCapture(VIDEO_PATH)
fps = int(cap.get(cv2.CAP_PROP_FPS)) if cap.get(cv2.CAP_PROP_FPS) > 0 else 25
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

bytetrack_args = ByteTrackArgs(fps=fps)
tracker = BYTETracker(bytetrack_args, frame_rate=fps)

# State Management variables
track_windows = collections.defaultdict(lambda: collections.deque(maxlen=50))
track_labels = {}
io_write_buffers = collections.defaultdict(list)
person_colors = {}

# Keep a record of the last successfully calculated keypoint structures for frame skipping
last_known_kpts = {}

CLASS_MAPPING = {0: "Walking", 1: "Running"}


def get_unique_color(track_id):
    """Generates a stable, random high-contrast color bounding box based on Track ID."""
    if track_id not in person_colors:
        np.random.seed(int(track_id))
        color = tuple(int(c) for c in np.random.randint(0, 255, size=3))
        person_colors[track_id] = color
    return person_colors[track_id]


print(f"Tracking Pipeline Operational: Res {width}x{height} @ {fps} FPS | Frame Skip Step: {FRAME_STEP}")

# ----------------------------------------------------
# MAIN MOVEMENT TRACKING PIPELINE
# ----------------------------------------------------
frame_idx = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_idx += 1

    # Initialize empty lists at the top of the loop to prevent carrying over dead data
    boxes, keypoints_list = [], []

    if frame_idx == 1 or (frame_idx % FRAME_STEP == 0):
        boxes, keypoints_list = process_frame(frame, pose_model, device)
        # Cache these globally for skipped frames
        last_boxes, last_kpts_list = boxes, keypoints_list
    else:
        # Use cached data so tracking and matching don't break
        boxes = last_boxes if 'last_boxes' in locals() else []
        keypoints_list = last_kpts_list if 'last_kpts_list' in locals() else []


    if len(boxes) > 0:
        detections = np.array(boxes, dtype=np.float32)
        online_targets = tracker.update(detections, (height, width), (height, width))
    else:
        online_targets = []

    for target in online_targets:
        tlwh = target.tlwh
        track_id = target.track_id

        # Spatial Bounding Box coordinate conversion [x1, y1, x2, y2]
        x1, y1, w, h = tlwh
        xyxy = [int(x1), int(y1), int(x1 + w), int(y1 + h)]

        # Map targets using centroid distance checks
        target_center = np.array([(x1 + w / 2), (y1 + h / 2)])
        best_match_idx = -1
        min_distance = float('inf')

        for idx, b in enumerate(boxes):
            box_center = np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2])
            dist = np.linalg.norm(target_center - box_center)
            if dist < min_distance:
                min_distance = dist
                best_match_idx = idx

        # Determine keypoint values to use
        if best_match_idx != -1 and len(keypoints_list) > 0:
            assigned_kpts = keypoints_list[best_match_idx]
            last_known_kpts[track_id] = assigned_kpts  # Cache for frame skipping
        elif track_id in last_known_kpts:
            assigned_kpts = last_known_kpts[track_id]  # Reuse cached coordinates on skipped frames
        else:
            continue

        # Generate 42-D Vector Signature
        embedding_42d = keypoints_to_embedding(assigned_kpts)

        # Append tracking records to memory buffer
        io_write_buffers[track_id].append(" ".join(map(str, embedding_42d)) + "\n")
        if len(io_write_buffers[track_id]) >= 25:
            txt_filename = os.path.join(OUTPUT_TXT_DIR, f"person_{track_id}.txt")
            with open(txt_filename, "a") as txt_file:
                txt_file.writelines(io_write_buffers[track_id])
            io_write_buffers[track_id].clear()

        # Update sliding history queue
        track_windows[track_id].append(embedding_42d)

        # BiLSTM Action Prediction Layer
        if len(track_windows[track_id]) == 50:
            input_sequence = np.array([list(track_windows[track_id])], dtype=np.float32)
            ort_outputs = ort_session.run(None, {input_name: input_sequence})

            # --- FIX FOR MATRIX DIMENSION FORMAT ERROR ---
            # Flatten the 2D output matrix [1, 2] into a clean 1D array row [2]
            probabilities = ort_outputs[0].flatten()

            class_id = int(np.argmax(probabilities))
            action_label = CLASS_MAPPING.get(class_id, "Unknown")

            # Formats correctly now because probabilities[class_id] evaluates to a single float scalar value
            track_labels[track_id] = f"{action_label} ({probabilities[class_id] * 100:.1f}%)"
        else:
            track_labels[track_id] = f"Buffering ({len(track_windows[track_id])}/50)"

        # ----------------------------------------------------
        # VISUALIZATION RENDERING PIPELINE
        # ----------------------------------------------------
        display_text = f"ID {track_id}: {track_labels[track_id]}"
        box_color = get_unique_color(track_id)
        plot_one_box(xyxy, frame, label=display_text, color=box_color, line_thickness=2)

    cv2.imshow("Multi-Person Action Tracking System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Flush remaining buffers when closing system
for track_id, lines in io_write_buffers.items():
    if lines:
        txt_filename = os.path.join(OUTPUT_TXT_DIR, f"person_{track_id}.txt")
        with open(txt_filename, "a") as txt_file:
            txt_file.writelines(lines)

cap.release()
cv2.destroyAllWindows()
print("Pipeline Stopped Successfully.")