# 🏃 Human Activity Recognition using Computer Vision

A **Computer Vision-based Human Activity Recognition (HAR)** system for classifying **suspicious** and **non-suspicious** activities from RGB videos. The project integrates **YOLOv7-W6 Pose**, **ByteTrack**, and a **2-layer BiLSTM** to perform real-time activity recognition through a Streamlit dashboard.

---

## 📖 Overview

This project detects and tracks people from RGB videos, extracts pose-based skeletal features, and performs temporal sequence modeling to classify activities in real time.

The pipeline combines **YOLOv7-W6 Pose** for human pose estimation, **ByteTrack** for multi-person tracking, and a **BiLSTM** network for temporal activity classification. The system currently recognizes activities such as **walking, running, drilling, and climbing**, which are further categorized into **suspicious** and **non-suspicious** activities.

---

## 🎥 Demo

The repository includes demonstration videos showcasing the complete pipeline, including:

- Human pose estimation
- Multi-person tracking
- Real-time activity classification
- Confidence score visualization
- Streamlit dashboard interface

---

## 🔄 Project Pipeline

```text
                RGB Video
                    │
                    ▼
      YOLOv7-W6 Pose Estimation
                    │
                    ▼
      ByteTrack Multi-Person Tracking
                    │
                    ▼
      Pose Keypoint Extraction
                    │
                    ▼
      42-Dimensional Pose Embedding
     (34 Coordinates + 8 Joint Angles)
                    │
                    ▼
      50-Frame Temporal Sequence
                    │
                    ▼
          2-Layer BiLSTM Model
                    │
                    ▼
   Suspicious / Non-Suspicious
      Activity Classification
                    │
                    ▼
     Streamlit Real-Time Dashboard
```

---

## ✨ Key Features

- Real-time **Human Activity Recognition**
- Pose estimation using **YOLOv7-W6 Pose**
- Multi-person tracking with **ByteTrack**
- Custom **42-dimensional pose embeddings**
- Temporal sequence modeling using **BiLSTM**
- ONNX model deployment
- Streamlit-based real-time inference dashboard
- Live confidence score visualization

---

## 🎯 Supported Activities

- 🚶 Walking
- 🏃 Running
- 🛠️ Drilling
- 🧗 Climbing

---

## 🦴 Pose Embedding

Each video frame is converted into a **42-dimensional embedding** consisting of:

### Coordinate Features (34)

- 17 body keypoints
- X and Y coordinates
- Torso-based normalization using shoulder and hip landmarks

### Joint Angle Features (8)

- Left Knee Flexion
- Right Knee Flexion
- Left Hip Flexion
- Right Hip Flexion
- Left Elbow Flexion
- Right Elbow Flexion
- Trunk Lean
- Inter-Thigh Angle

All coordinate values are normalized relative to the torso, while angular features are normalized by **180°**, resulting in feature values between **0 and 1**.

---

## 🧠 Model Architecture

```
Input Sequence (50 × 42)
          │
          ▼
   BiLSTM Layer (64)
          │
          ▼
   BiLSTM Layer (32)
          │
          ▼
    Dropout (0.2)
          │
          ▼
   Dense Layer (ReLU)
          │
          ▼
 Softmax Output (2 Classes)
```

---

## 📊 Performance

| Metric | Value |
|---------|------:|
| Overall Accuracy | **~96%** |

The final Human Activity Recognition system classifies suspicious and non-suspicious activities in real time using pose estimation, multi-person tracking, and temporal sequence modeling.

---

## 🛠️ Tech Stack

### Languages

- Python

### Frameworks & Libraries

- PyTorch
- TensorFlow / Keras
- OpenCV
- NumPy
- Scikit-learn
- Streamlit
- ONNX Runtime

### Models

- YOLOv7-W6 Pose
- ByteTrack
- BiLSTM

---

## 📂 Repository Structure

```text
Keypoints/
│
├── Dataset/
├── Output/
├── Person_Embeddings/
│
└── yolov7-main/
    │
    ├── ByteTrack-main/
    ├── cfg/
    ├── data/
    ├── debug_select/
    ├── deploy/
    ├── models/
    ├── scripts/
    ├── tools/
    ├── utils/
    ├── vids/
    │
    ├── app.py                     # Streamlit dashboard
    ├── tracker.py                 # ByteTrack implementation
    ├── kpt_detection.py           # Pose detection
    ├── extract_video_embeddings.py
    ├── extract_rem_video_embs.py
    ├── find_embeddings.py
    ├── training_bilstm.py
    ├── test_model.py
    ├── bilstm_model.onnx
    ├── yolov7-w6-pose.pt
    ├── requirements.txt
    ├── README.md
    │
    ├── Streamlit demo 1.MOV
    ├── Streamlit demo 2.MOV
    ├── Final Presentation.pptx
    └── Report.docx
```

---

## ⚙️ Installation

Clone the repository:

```bash
git clone <repository-url>
cd yolov7-main
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Launch the Streamlit application:

```bash
streamlit run app.py
```

---

## 📌 Disclaimer

This repository contains work completed during my **Machine Learning Internship at Prisma AI**.

Certain datasets, trained model weights, and portions of the implementation have been omitted or modified due to confidentiality requirements.
