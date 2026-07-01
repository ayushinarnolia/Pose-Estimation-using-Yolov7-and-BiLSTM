#test_model.py
import numpy as np
import onnxruntime as ort
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from dataset import load_dataset

# 1. Load dataset
X, y = load_dataset(r"C:\Users\ayush\OneDrive\Desktop\Internship\Keypoints\Output")

# 2. Load ONNX model
session = ort.InferenceSession(r"C:\Users\ayush\OneDrive\Desktop\Internship\Keypoints\yolov7-main\bilstm_model.onnx")

input_name = session.get_inputs()[0].name

# 3. Run prediction
y_pred = []

for sample in X:
    sample = sample.astype(np.float32)
    sample = np.expand_dims(sample, axis=0)  # (1, 50, 42)

    pred = session.run(None, {input_name: sample})
    y_pred.append(np.argmax(pred[0]))

y_pred = np.array(y_pred)

# 4. Evaluation
print("Accuracy:", accuracy_score(y, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y, y_pred))
print("\nClassification Report:\n", classification_report(y, y_pred))