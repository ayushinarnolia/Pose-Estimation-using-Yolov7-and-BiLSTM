#model.py
# Defines and compiles the LSTM-based neural network model
# Input: sequence data (50 timesteps × 42 features)
# Output: probability of classes (Walking / Running)
# Uses stacked LSTM layers + Dense layers for classification

from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Bidirectional, Input

def build_model():
    model = Sequential()

    # Layer 1
    # model.add(Bidirectional(LSTM(64, return_sequences=True, input_shape=(50, 42))))
    model.add(Input(shape=(50, 42)))
    model.add(Bidirectional(LSTM(64, return_sequences=True)))

    # Layer 2
    model.add(Bidirectional(LSTM(32)))

    model.add(Dropout(0.2))

    model.add(Dense(32, activation='relu'))

    # Output layer (2 classes now, scalable later)
    model.add(Dense(2, activation='softmax'))

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model