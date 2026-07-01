#video.py
import cv2
import numpy as np
import time
import torch
from kpt_detection import load_kpt_model, predict_kpt
from find_embeddings import keypoints_to_embedding

# --- Video and model paths ---
video_path = r"C:\Users\ayush\OneDrive\Desktop\Internship\Keypoints\yolov7-main\walking_demo.webm"
model_path = r"C:\Users\ayush\OneDrive\Desktop\Internship\Keypoints\yolov7-main\yolov7-w6-pose.pt"

# --- Load video and keypoint model ---
cap = cv2.VideoCapture(video_path)
device, model = load_kpt_model(model_path)

frame_count = 0
output_frame = None

all_keypoints = []   # per-frame keypoints (list of [[x,y,conf], ...])
all_embeddings = []  # per-frame embeddings (list of 42 floats)

# --- FPS control ---
TARGET_FPS = 25
actual_fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"Original Video FPS: {actual_fps}")
print(f"Total frames in video: {total_frames}")

if actual_fps > TARGET_FPS:
    frame_skip = max(1, round(actual_fps / TARGET_FPS))
else:
    frame_skip = 1

start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret or frame_count >= total_frames:
        print("Video reached the end successfully.")
        break

    frame_count += 1
    if frame_count == 16:
        break
    # --- Only process frames according to FPS target ---
    if frame_count == 1 or frame_count % frame_skip == 0:

        h, w = frame.shape[:2]
        target_w, target_h = 640, 360
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)

        resized_frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

        pad_w = target_w - new_w
        pad_h = target_h - new_h

        resized_frame = cv2.copyMakeBorder(
            resized_frame,
            top=pad_h // 2,
            bottom=pad_h - pad_h // 2,
            left=pad_w // 2,
            right=pad_w - pad_w // 2,
            borderType=cv2.BORDER_CONSTANT,
            value=[0, 0, 0]
        )

        # --- Keypoint prediction ---
        kpts, output_frame = predict_kpt(resized_frame, device, model)

        frame_persons = []
        embedding = [0.0] * 42  # default embedding if no person detected

        num_persons = kpts.shape[0]

        if num_persons > 0:
            primary_person = kpts[0]

            raw_kpts = primary_person[7:]  # remove bbox info
            kpts_reshaped = raw_kpts.reshape(-1, 3)
            frame_persons = kpts_reshaped.tolist()

            if len(frame_persons) > 0:
                keypoints_xy = [[kp[0], kp[1]] for kp in frame_persons]
                embedding = keypoints_to_embedding(keypoints_xy)
                embedding = list(embedding)  # convert np.array to Python list
                if len(embedding) != 42:
                    print(f"[WARNING] embedding length != 42, forcing zeros")
                    embedding = [0.0] * 42

        # --- Append results ---
        all_keypoints.append(frame_persons)
        all_embeddings.append(embedding)

    else:
        # Skipped frames: keep alignment
        all_keypoints.append([])
        all_embeddings.append([0.0] * 42)

        if output_frame is None:
            output_frame = cv2.resize(frame, (640, 360), interpolation=cv2.INTER_AREA)

    # --- Display (optional) ---
    cv2.imshow('demo', output_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

end_time = time.time()
elapsed = end_time - start_time

print(f"\nProcessed {frame_count} frames in {elapsed:.2f} seconds ({frame_count / elapsed:.2f} FPS).")
print("\nFinal list of all keypoints per processed frame:")
print(all_keypoints)
print("\nFinal list of all embeddings (list of lists):")
print(all_embeddings[:5])  # print first 5 frames as sample
print("Number of frames:", len(all_embeddings))
print("Length of first frame embedding:", len(all_embeddings[0]))

# --- Save embeddings as npy (list-of-lists format) ---
np.save("video_embeddings.npy", all_embeddings)

cap.release()
cv2.destroyAllWindows()