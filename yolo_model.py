
import os
import torch
from utils.utils import capture_screen
from ultralytics import YOLO
from win32api import SetCursorPos
import numpy as np
import pyautogui
from quantum.qcnn import QuantumNeuralNet
import time

class YOLODetection:
    def __init__(self, use_gpu=False):
        self.device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
        self.model = YOLO('yolov8n.pt').to(self.device)  # Using a smaller YOLOv8n model for quick detection
        self.qnn = QuantumNeuralNet().to(self.device)    # Load the Quantum Neural Network
        self.sensitivity = 1.0
        self.smooth_factor = 1.0

    def detect_and_process(self, frame):
        results = self.model(frame)
        predictions = results.xyxy[0].cpu().numpy()  # Get the predictions

        if len(predictions) > 0:
            for pred in predictions:
                x_center = int((pred[0] + pred[2]) / 2)
                y_center = int((pred[1] + pred[3]) / 2)
                self.process_and_move_cursor(x_center, y_center)

    def process_and_move_cursor(self, x, y):
        # Use the quantum network to refine predictions
        q_input = np.array([x, y])
        q_output = self.qnn.forward(torch.tensor(q_input, device=self.device).float())
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

if __name__ == "__main__":
    use_gpu = torch.cuda.is_available()
    yolo = YOLODetection(use_gpu=use_gpu)
    
    while True:
        frame = capture_screen({"top": 100, "left": 100, "width": 800, "height": 600})
        yolo.detect_and_process(frame)
        time.sleep(0.01)  # Small delay to prevent overloading the CPU/GPU
