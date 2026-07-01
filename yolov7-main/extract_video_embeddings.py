#extract_video_embeddings.py
import os
import cv2
import gc
import torch
import warnings
import numpy as np
from tqdm import tqdm

from kpt_detection import load_kpt_model, predict_kpt
from find_embeddings import keypoints_to_embedding

warnings.filterwarnings("ignore")
torch.set_printoptions(profile="default")

# ----------------------------
# LOAD MODEL
# ----------------------------
def initialize_model(model_path):
    device, model = load_kpt_model(model_path)
    model.eval()
    return device, model


# ----------------------------
# DRAW BOXES
# ----------------------------

def draw_boxes(frame, kpts):
    img = frame.copy()
    for i, p in enumerate(kpts):
        x1, y1, x2, y2 = map(int, p[2:6])
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            img,
            f"ID {i}",
            (x1, max(20, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )
    return img


# ----------------------------
# SELECT PERSON (SERVER SAFE)
# ----------------------------
def select_main_person(frame, kpts, video_name):
    display = draw_boxes(frame, kpts)

    os.makedirs("debug_select", exist_ok=True)

    path = f"debug_select/{video_name}_select.jpg"

    cv2.imwrite(path, display)

    print("\n" + "=" * 60)
    print("PERSON SELECTION IMAGE SAVED:")
    print(path)
    print("Open this image and choose ID visually.")
    print("=" * 60)

    while True:
        idx = input(f"Enter MAIN PERSON ID (0-{len(kpts)-1}): ")

        if idx.isdigit():
            idx = int(idx)
            if 0 <= idx < len(kpts):
                return idx

        print("Invalid input")


# ----------------------------
# EMBEDDING EXTRACTION
# ----------------------------
def extract_embedding(frame, device, model, main_id=None, video_name="video"):

    with torch.no_grad():
        kpts, _, _ = predict_kpt(frame, device, model)

    if kpts is None or len(kpts) == 0:
        return None, main_id

    if main_id is None and len(kpts) > 1:
        main_id = select_main_person(frame, kpts, video_name)

    if main_id is None:
        main_id = 0

    if main_id >= len(kpts):
        main_id = 0

    person = kpts[main_id]

    raw = person[7:]

    if len(raw) % 3 != 0:
        return None, main_id

    kp = raw.reshape(-1, 3)
    xy = kp[:, :2]

    embedding = keypoints_to_embedding(xy)

    embedding = np.array(embedding, dtype=np.float32).flatten()

    if len(embedding) != 42:
        return None, main_id

    return embedding.tolist(), main_id


# ----------------------------
# PROCESS VIDEO
# ----------------------------
def process_video(video_path, device, model, num_embeddings=50):

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Cannot open:", video_path)
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0 or frame_count <= 0:
        return None

    num_candidates = int(num_embeddings * 2.5)
    indices = set(np.linspace(0, frame_count - 1, num_candidates, dtype=int))

    embeddings = []
    main_id = None

    frame_idx = 0
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    while True:

        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx in indices:

            emb, main_id = extract_embedding(
                frame,
                device,
                model,
                main_id,
                video_name
            )

            if emb is not None:
                embeddings.append(emb)

            if len(embeddings) >= num_embeddings:
                break

        frame_idx += 1

    cap.release()

    return embeddings


# ----------------------------
# PROCESS DATASET
# ----------------------------
def process_dataset(dataset_path, output_path, model_path):

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

        os.makedirs(out_dir, exist_ok=True)

        out_file = os.path.join(
            out_dir,
            os.path.splitext(v)[0] + ".txt"
        )

        embeddings = process_video(
            in_path,
            device,
            model
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
    output_path = r"C:\Users\ayush\OneDrive\Desktop\Internship\Keypoints\Dataset output"
    model_path = r"yolov7-w6-pose.pt"

    process_dataset(dataset_path, output_path, model_path)