
import time
import torch
import tensorflow as tf
from utils.utils import capture_screen
from keras import Sequential
from keras import Conv2D, MaxPooling2D, BatchNormalization, ReLU, Dropout, Flatten, Dense
from quantum.qcnn import QuantumNeuralNet
from win32 import SetCursorPos
import numpy as np
import pyautogui

class FaceRecognition:
    def __init__(self, model_path, use_gpu=False):
        self.device = "/gpu:0" if use_gpu and tf.config.list_physical_devices('GPU') else "/cpu:0"
        with tf.device(self.device):
            self.model = self.build_model()
        self.qnn = QuantumNeuralNet().to(torch.device("cuda" if use_gpu else "cpu"))  # Quantum Neural Network
        self.sensitivity = 1.0
        self.smooth_factor = 1.0

    def build_model(self):
        model = Sequential([
            Conv2D(32, (3, 3), padding='same', input_shape=(64, 64, 3)),
            ReLU(),
            MaxPooling2D(),
            Conv2D(64, (3, 3), padding='same'),
            ReLU(),
            MaxPooling2D(),
            Flatten(),
            Dense(128),
            ReLU(),
            Dropout(0.5),
            Dense(2, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model

    def detect_and_process(self, frame):
        with tf.device(self.device):
            predictions = self.model.predict(frame)
        for pred in predictions:
            x_center = int(pred[0] * frame.shape[1])
            y_center = int(pred[1] * frame.shape[0])
            self.process_and_move_cursor(x_center, y_center)

    def process_and_move_cursor(self, x, y):
        # Use the quantum network to refine predictions
        q_input = np.array([x, y])
        q_output = self.qnn.forward(torch.tensor(q_input, device=self.qnn.device).float())
        q_output = q_output.detach().cpu().numpy()

        # Apply smoothing to the cursor movement
        current_pos = pyautogui.position()
        new_x = int(current_pos[0] + (q_output[0] - current_pos[0]) * self.smooth_factor)
        new_y = int(current_pos[1] + (q_output[1] - current_pos[1]) * self.smooth_factor)
        SetCursorPos((new_x, new_y))

    def set_sensitivity(self, sensitivity):
        self.sensitivity = sensitivity

    def set_smooth_factor(self, smooth_factor):
        self.smooth_factor = smooth_factor

    def save_model(self, path):
        self.model.save(path)

    def load_model(self, path):
        self.model = tf.keras.models.load_model(path)

if __name__ == "__main__":
    use_gpu = tf.config.list_physical_devices('GPU')
    face_recog = FaceRecognition(model_path='facial_model.h5', use_gpu=use_gpu)
    
    while True:
        frame = capture_screen({"top": 100, "left": 100, "width": 800, "height": 600})
        face_recog.detect_and_process(frame)
        time.sleep(0.01)  # Small delay to prevent overloading the CPU/GPU
