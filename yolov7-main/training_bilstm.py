#training_bilstm.py
# Main training script for the LSTM model
# Loads dataset, splits into train/test sets
# Builds model, trains it using training data
# Uses EarlyStopping to prevent overfitting
# Saves trained model as .h5 file for later use

from dataset import load_dataset
from model import build_model
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
import tf2onnx
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# Load data
X, y = load_dataset(r"C:\Users\ayush\OneDrive\Desktop\Internship\Keypoints\Output")

print("X shape:", X.shape)  # (num_samples, 50, 42)

# rain-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)

# Build model
model = build_model()
model.summary()

# Early stopping
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=15,
    restore_best_weights=True
)

# Train
history = model.fit(
    X_train, y_train,
    epochs=50000,
    batch_size=8,
    validation_data=(X_test, y_test),
    callbacks=[early_stop]
)

# Save model
#model.save("bilstm_model.h5")
spec = (tf.TensorSpec((None, 50, 42), tf.float32, name="input"),)

onnx_model, _ = tf2onnx.convert.from_keras(
    model,
    input_signature=spec,
    output_path="bilstm_model.onnx"
)

print("Model saved as ONNX!")
print("Training complete!")

#Predictions:
#
# video = load_video("test.txt")
# video = video.reshape(1, 50, 42)
#
# pred = model.predict(video)
#
# print(np.argmax(pred))