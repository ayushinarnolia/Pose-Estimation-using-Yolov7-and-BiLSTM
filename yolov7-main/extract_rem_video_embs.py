# extract_rem_vids_embs.py
import os
import cv2
import gc
import torch
import warnings
import shutil
import numpy as np
from tqdm import tqdm
from torchvision import transforms

# --- Import official ByteTrack from your cloned repository ---
try:
    from yolox.tracker.byte_tracker import BYTETracker
except ImportError:
    import sys

    sys.path.append(r"ByteTrack-main")
    from yolox.tracker.byte_tracker import BYTETracker


from utils.datasets import letterbox
from utils.general import non_max_suppression_kpt, scale_coords
from find_embeddings import keypoints_to_embedding

warnings.filterwarnings("ignore")
torch.set_printoptions(profile="default")


# Mock argument payload expected by the native ByteTrack class
class TrackerArgs:
    def __init__(self):
        self.track_thresh = 0.25  # Detection score threshold
        self.track_buffer = 30  # Frames to keep lost tracks active
        self.match_thresh = 0.8  # IOU matching threshold
        self.mot20 = False  # MOT20 execution flag


# ----------------------------
# LOAD MODEL (From your kpt.py)
# ----------------------------
def initialize_model(model_path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    weights = torch.load(model_path, map_location=device, weights_only=False)
    model = weights['model']
    _ = model.float().eval()

    if torch.cuda.is_available():
        model.half().to(device)
    return device, model


# ----------------------------
# PROCESS FRAME (Your exact kpt.py function)
# ----------------------------
def process_frame(frame, model, device):
    img_padded = letterbox(frame, 960, stride=64, auto=True)[0]
    img_tensor = transforms.ToTensor()(img_padded)
    img_tensor = torch.tensor(np.array([img_tensor.numpy()]))

    if torch.cuda.is_available():
        img_tensor = img_tensor.half().to(device)

    with torch.no_grad():
        output, _ = model(img_tensor)

    output = non_max_suppression_kpt(
        output, 0.25, 0.65,
        nc=model.yaml['nc'],
        nkpt=model.yaml['nkpt'],
        kpt_label=True
    )

    boxes = []
    keypoints_list = []

    for det in output:
        if len(det):
            scale_coords(img_tensor.shape[2:], det[:, :4], frame.shape)

            kpts_all = det[:, 6:]
            num_people = kpts_all.shape[0]
            kpts_reshaped = kpts_all.view(num_people, 17, 3)

            gain = min(img_tensor.shape[2] / frame.shape[0], img_tensor.shape[3] / frame.shape[1])
            pad_x = (img_tensor.shape[3] - frame.shape[1] * gain) / 2
            pad_y = (img_tensor.shape[2] - frame.shape[0] * gain) / 2

            kpts_scaled_list = []
            for p in range(num_people):
                person_kpts = kpts_reshaped[p]
                xy_coords = person_kpts[:, :2].clone()
                xy_coords[:, 0] = (xy_coords[:, 0] - pad_x) / gain
                xy_coords[:, 1] = (xy_coords[:, 1] - pad_y) / gain
                conf_scores = person_kpts[:, 2:3]
                scaled_person = torch.cat([xy_coords, conf_scores], dim=1)
                kpts_scaled_list.append(scaled_person)

            if num_people > 0:
                kpts_recombined = torch.stack(kpts_scaled_list, dim=0)
                det[:, 6:] = kpts_recombined.view(num_people, -1)

            det = det.detach().cpu().numpy()

            for d in det:
                boxes.append([
                    int(round(d[0])),
                    int(round(d[1])),
                    int(round(d[2])),
                    int(round(d[3])),
                    float(d[4])
                ])
                kpts_raw = d[6:]
                individual_kpts = kpts_raw.reshape(-1, 3)
                xy_only = individual_kpts[:, :2]
                xy_only_int = np.round(xy_only).astype(np.int32)
                keypoints_list.append(xy_only_int)

    return boxes, keypoints_list


# ----------------------------
# DRAW INTERACTIVE BOXES (From original script)
# ----------------------------
def draw_boxes(frame, online_targets):
    img = frame.copy()
    for target in online_targets:
        track_id = target.track_id
        x1, y1, x2, y2 = map(int, target.tlbr)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            img,
            f"Track ID {track_id}",
            (x1, max(20, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )
    return img


# ----------------------------
# INTERACTIVE SELECTION (From original script)
# ----------------------------
def select_main_person(frame, online_targets, video_name):
    display = draw_boxes(frame, online_targets)
    os.makedirs("debug_select", exist_ok=True)
    path = f"debug_select/{video_name}_select.jpg"
    cv2.imwrite(path, display)

    print("\n" + "=" * 60)
    print("TRACKED PERSON SELECTION IMAGE SAVED:")
    print(path)
    print("Open this image and verify the Track ID.")
    print("=" * 60)

    valid_ids = [t.track_id for t in online_targets]
    while True:
        idx = input(f"Enter MAIN PERSON TRACK ID ({valid_ids}): ")
        if idx.isdigit():
            idx = int(idx)
            if idx in valid_ids:
                return idx
        print("Invalid input choice.")


# ----------------------------
# PROCESS VIDEO WITH BYTE-KALMAN TRACKER
# ----------------------------
def process_video(video_path, device, model, skip_folder, num_embeddings=50):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Cannot open:", video_path)
        return None

    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0 or frame_count <= 0:
        return None

    # Initialize ByteTrack Tracking Instance
    args = TrackerArgs()
    tracker = BYTETracker(args, frame_rate=fps)

    embeddings = []
    video_name = os.path.basename(video_path)
    video_base_name = os.path.splitext(video_name)[0]

    target_track_id = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        boxes, keypoints_list = process_frame(frame, model, device)

        if not boxes:
            # Maintain inner Kalman filter matrices even during blank frames
            online_targets = tracker.update(np.empty((0, 5)), [height, width], [height, width])
            continue

        output_results = np.array(boxes, dtype=np.float32)
        online_targets = tracker.update(output_results, [height, width], [height, width])

        if len(online_targets) == 0:
            continue

        # Handle Target ID Allocation Context
        if target_track_id is None:
            if len(online_targets) == 1:
                # If only one person exists, automatically lock on to their ID
                target_track_id = online_targets[0].track_id
            else:
                # Interactive option: If multiple people are seen, choose visually
                target_track_id = select_main_person(frame, online_targets, video_base_name)

        # Process frame embeddings if our tracked target object is alive inside the current frame
        matched_target = [t for t in online_targets if t.track_id == target_track_id]

        if matched_target:
            target = matched_target[0]
            t_box = target.tlbr  # [x1, y1, x2, y2] format

            # Map tracking box back to the closest index in raw predictions list
            best_idx = 0
            min_dist = float('inf')

            for idx, b in enumerate(boxes):
                cx_b, cy_b = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
                cx_t, cy_t = (t_box[0] + t_box[2]) / 2, (t_box[1] + t_box[3]) / 2
                dist = np.hypot(cx_b - cx_t, cy_b - cy_t)
                if dist < min_dist:
                    min_dist = dist
                    best_idx = idx

            xy = keypoints_list[best_idx]
            embedding = keypoints_to_embedding(xy)
            embedding = np.array(embedding, dtype=np.float32).flatten()

            if len(embedding) == 42:
                embeddings.append(embedding.tolist())

        if len(embeddings) >= num_embeddings:
            break

    cap.release()
    return embeddings


# ----------------------------
# PROCESS DATASET
# ----------------------------
def process_dataset(dataset_path, output_path, skip_base_path, model_path):
    device, model = initialize_model(model_path)
    os.makedirs(output_path, exist_ok=True)

    videos = []
    for cls in os.listdir(dataset_path):
        folder = os.path.join(dataset_path, cls)
        if not os.path.isdir(folder):
            continue
        for v in os.listdir(folder):
            if v.endswith((".mp4", ".avi", ".mov", ".webm")):
                videos.append((cls, v))

    print("Total videos:", len(videos))

    for cls, v in tqdm(videos):
        in_path = os.path.join(dataset_path, cls, v)
        out_dir = os.path.join(output_path, cls)
        skip_folder = os.path.join(skip_base_path, cls)

        os.makedirs(out_dir, exist_ok=True)
        out_file = os.path.join(out_dir, os.path.splitext(v)[0] + ".txt")

        embeddings = process_video(
            in_path,
            device,
            model,
            skip_folder
        )

        if not embeddings:
            continue

        np.savetxt(out_file, np.array(embeddings), fmt="%.6f")
        gc.collect()


# --------------------------------------------------
# MAIN ENTRY
# --------------------------------------------------
if __name__ == "__main__":
    dataset_path = r"C:\Users\ayush\OneDrive\Desktop\Internship\Keypoints\Dataset"
    output_path = r"C:\Users\ayush\OneDrive\Desktop\Internship\Keypoints\Output"
    skip_path = r"C:\Users\ayush\OneDrive\Desktop\Internship\Keypoints\Dataset skipped multiperson"
    model_path = r"yolov7-w6-pose.pt"

    process_dataset(dataset_path, output_path, skip_path, model_path)