#app.py
# ----------------------------------------------------
# STREAMLIT APP - OPENCV VIDEO PLAYBACK VERSION
# ----------------------------------------------------
import os
import cv2
import numpy as np
import collections
import torch
import onnxruntime as ort
import streamlit as st
import time
import pandas as pd
from kpt import load_model, process_frame
from find_embeddings import keypoints_to_embedding
from utils.plots import plot_one_box
import tempfile
from pathlib import Path

# ----------------------------------------------------
# CPU SETUP
# ----------------------------------------------------
device = torch.device("cpu")
torch.set_num_threads(4)
# ----------------------------------------------------
# BYTE TRACK
# ----------------------------------------------------
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
# CONFIG
# ----------------------------------------------------
WEIGHTS_PATH = r'yolov7-w6-pose.pt'
ONNX_MODEL_PATH = r'bilstm_model.onnx'
OUTPUT_TXT_DIR = r'C:\Users\ayush\OneDrive\Desktop\Internship\Keypoints\Person_Embeddings'
FRAME_STEP = 2
os.makedirs(OUTPUT_TXT_DIR, exist_ok=True)
CLASS_MAPPING = {0: "Walking", 1: "Running"}
# ----------------------------------------------------
# UI
# ----------------------------------------------------
st.set_page_config(page_title="Action Tracking System", layout="wide")
st.title("🏃 Multi-Person Action Tracking & LIVE OpenCV Playback")
st.sidebar.header("Video Configuration")

uploaded_video = st.sidebar.file_uploader(
    "📁 Upload a Video",
    type=[
        "mp4","avi","mov","mkv","mpeg","mpg",
        "webm","wmv","m4v","flv","3gp",
        "ts","m2ts","mts","asf","ogv"
    ],
    help="Drag & drop or browse to upload a video."
)

if uploaded_video is not None:
    st.subheader("Uploaded Video")
    st.video(uploaded_video)

run_button = st.sidebar.button("Run Detection & Classification")
status_placeholder = st.empty()
# 🔥 OpenCV playback placeholder
frame_placeholder = st.empty()

# ✅ NEW: table placeholder
table_placeholder = st.empty()

track_colors = {}
def get_color(track_id):
    if track_id not in track_colors:
        np.random.seed(int(track_id))  # ensures same ID → same color
        track_colors[track_id] = tuple(int(c) for c in np.random.randint(0, 255, 3))
    return track_colors[track_id]
# ----------------------------------------------------
# RUN
# ----------------------------------------------------
if uploaded_video is not None and run_button:

    # Save uploaded file temporarily
    suffix = Path(uploaded_video.name).suffix

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        tmp_file.write(uploaded_video.read())
        video_path_input = tmp_file.name

    status_placeholder.info("Loading models...")
    device, pose_model = load_model(WEIGHTS_PATH)
    ort_session = ort.InferenceSession(
        ONNX_MODEL_PATH,
        providers=["CPUExecutionProvider"]
    )
    input_name = ort_session.get_inputs()[0].name
    cap = cv2.VideoCapture(video_path_input)

    if not cap.isOpened():
        st.error("Unable to open the uploaded video.")
        if os.path.exists(video_path_input):
            os.remove(video_path_input)
        st.stop()
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = int(fps) if fps and fps > 1 else 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    tracker = BYTETracker(ByteTrackArgs(fps=fps), frame_rate=fps)
    track_windows = collections.defaultdict(lambda: collections.deque(maxlen=50))
    track_labels = {}
    last_kpts = {}
    last_boxes = []
    last_keypoints = []
    frame_idx = 0
    status_placeholder.success("Processing + Live playback started...")
    # ------------------------------------------------
    # MAIN LOOP (REAL-TIME DISPLAY)
    # ------------------------------------------------
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        if frame is None or frame.size == 0:
            continue
        # Pose inference
        if frame_idx == 1 or frame_idx % FRAME_STEP == 0:
            boxes, keypoints_list = process_frame(frame, pose_model, device)
            last_boxes, last_keypoints = boxes, keypoints_list
        else:
            boxes, keypoints_list = last_boxes, last_keypoints
        # Tracking
        if len(boxes) > 0:
            detections = np.array(boxes, dtype=np.float32)
            online_targets = tracker.update(detections, (height, width), (height, width))
        else:
            online_targets = []
        # Process tracks
        for target in online_targets:
            tlwh = target.tlwh
            track_id = target.track_id
            x1, y1, w, h = tlwh
            xyxy = [int(x1), int(y1), int(x1 + w), int(y1 + h)]
            center = np.array([x1 + w / 2, y1 + h / 2])
            best_idx = -1
            min_dist = float("inf")
            for i, b in enumerate(boxes):
                b_center = np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2])
                dist = np.linalg.norm(center - b_center)
                if dist < min_dist:
                    min_dist = dist
                    best_idx = i
            if best_idx != -1 and len(keypoints_list) > 0:
                kpts = keypoints_list[best_idx]
                last_kpts[track_id] = kpts
            elif track_id in last_kpts:
                kpts = last_kpts[track_id]
            else:
                continue
            embedding = keypoints_to_embedding(kpts)
            track_windows[track_id].append(embedding)
            if len(track_windows[track_id]) == 50:
                seq = np.array([list(track_windows[track_id])], dtype=np.float32)
                out = ort_session.run(None, {input_name: seq})
                probs = out[0].flatten()
                cls = int(np.argmax(probs))
                track_labels[track_id] = CLASS_MAPPING.get(cls, "Unknown")
            else:
                track_labels[track_id] = "Buffering"
            text = f"ID {track_id}: {track_labels[track_id]}"
            color = get_color(track_id)
            plot_one_box(xyxy, frame, label=text, color=color, line_thickness=2)

        # ------------------------------------------------
        # 🔥 OPEN-CV DISPLAY INSIDE STREAMLIT
        # ------------------------------------------------
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(
            frame_rgb,
            channels="RGB",
            use_container_width=True
        )

        # ✅ NEW: live table update
        df = pd.DataFrame(list(track_labels.items()), columns=["ID", "Class"])
        table_placeholder.dataframe(df, use_container_width=True)

    cap.release()
    cv2.destroyAllWindows()

    if os.path.exists(video_path_input):
        os.remove(video_path_input)

    status_placeholder.success("Playback finished!")
else:
    status_placeholder.info("📁 Upload a video and click 'Run Detection & Classification'.")


# to run: python -m streamlit run app.py --server.maxUploadSize=200 --server.maxMessageSize=200