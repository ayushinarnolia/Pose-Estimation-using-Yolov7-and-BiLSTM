#dataset.py
# Loads and preprocesses dataset from .txt files
# Each file represents a video with 50 frames and 42 features per frame
# Converts folder structure (Walking/Running) into:
# X → (num_samples, 50, 42)
# y → class labels (0 = Walking, 1 = Running)

import os
import numpy as np

def load_video(file_path):
    data = []

    with open(file_path, 'r') as f:
        for line in f:
            values = list(map(float, line.strip().split()))
            data.append(values)

    data = np.array(data)

    # Ensure correct shape
    if data.shape[0] < 50:
        # PAD with zeros
        pad = np.zeros((50 - data.shape[0], 42))
        data = np.vstack((data, pad))

    elif data.shape[0] > 50:
        # TRUNCATE
        data = data[:50]

    return data


def load_dataset(base_path):
    X = []
    y = []

    classes = {"Walking": 0, "Running": 1}

    for label_name, label in classes.items():
        folder = os.path.join(base_path, label_name)

        for file in os.listdir(folder):
            if file.endswith(".txt"):
                path = os.path.join(folder, file)

                video = load_video(path)

                X.append(video)
                y.append(label)

    return np.array(X), np.array(y)