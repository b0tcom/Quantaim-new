import time
from xml.parsers.expat import model
import matplotlib.pyplot as plt
import ctypes
import numpy as np
import torch
try:
    import pyautogui
except ImportError:
    print("pyautogui module not found. Please install it using 'pip install pyautogui'.")
import torch.nn as nn
import torch.optim as optim
from utils.utils import cv2_imshow
from facial_recognition import get_position, get_timestamp, get_velocity
from dataclasses import AimData
import qcnn as quantum_neuralnet # Import the quantum neural network module
import pennylane as qml

def collect_data():
    # Collect aim assist data
    position = get_position()
    velocity = get_velocity()
    timestamp = get_timestamp()
    data = AimData(position, velocity, timestamp)
    quantum_neuralnet.process_data(data)

class AimAssist:
    def __init__(self, sensitivity=1.0, smooth_factor=1.0, fov=90, activation_key='shift'):
        self.sensitivity = sensitivity
        self.smooth_factor = smooth_factor
        self.fov = fov
        self.activation_key = activation_key
        self.qnn = QuantumNeuralNet().to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    def is_key_pressed(self):
        return pyautogui.keyDown(self.activation_key)

    def move_cursor(self, x, y):
        pyautogui.moveTo(x * self.sensitivity, y * self.sensitivity)

    def click_target(self, button='left'):
        pyautogui.click(button=button)

    def aim_at_target(self, x1, y1, x2, y2):
        x_center = (x1 + x2) // 2
        y_center = (y1 + y2) // 2
        self.move_cursor(x_center, y_center)
        self.click_target()

    def calculate_distance(self, x1, y1, x2, y2):
        return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    def smooth_aim(self, current_pos, target_pos):
        x_current, y_current = current_pos
        x_target, y_target = target_pos
        steps = int(self.calculate_distance(x_current, y_current, x_target, y_target) / self.smooth_factor)
        if steps == 0:
            return
        x_step = (x_target - x_current) / steps
        y_step = (y_target - y_current) / steps
        for _ in range(steps):
            pyautogui.moveRel(x_step, y_step)
            time.sleep(0.01)

    def target_within_fov(self, detection):
        screen_width, screen_height = pyautogui.size()
        x_center_screen = screen_width // 2
        y_center_screen = screen_height // 2
        x1, y1, x2, y2 = detection['xmin'], detection['ymin'], detection['xmax'], detection['ymax']
        x_center_target = (x1 + x2) // 2
        y_center_target = (y1 + y2) // 2
        distance = self.calculate_distance(x_center_screen, y_center_screen, x_center_target, y_center_target)
        fov_radius = (self.fov / 180.0) * screen_width
        return distance <= fov_radius

    def target_closest_in_fov(self, detections):
        if not detections or not self.is_key_pressed():
            return
        closest_distance = float('inf')
        closest_object = None
        for detection in detections:
            if self.target_within_fov(detection):
                x1, y1, x2, y2 = detection['xmin'], detection['ymin'], detection['xmax'], detection['ymax']
                distance = self.calculate_distance(x1, y1, x2, y2)
                if distance < closest_distance:
                    closest_distance = distance
                    closest_object = detection
        if closest_object:
            x1, y1, x2, y2 = closest_object['xmin'], closest_object['ymin'], closest_object['xmax'], closest_object['ymax']
            self.aim_at_target(x1, y1, x2, y2)

    def train_qnn(self, train_loader, num_epochs=10, learning_rate=0.001):
        self.qnn.train_model(train_loader, num_epochs, learning_rate)

    def evaluate_qnn(self, test_loader):
        self.qnn.evaluate_model(test_loader)

class QuantumNeuralNet(nn.Module):
    def __init__(self):
        super(QuantumNeuralNet, self).__init__()
        self.quantum_weights = torch.nn.Parameter(0.01 * torch.randn(3, 4, 3))
        self.fc1 = nn.Linear(4, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.to(self.quantum_weights.device)
        q_out = quantum_conv_circuit(x, self.quantum_weights)
        q_out = torch.tensor(q_out, device=x.device)
        x = torch.relu(self.fc1(q_out))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.softmax(x)
        return x

    def train_model(self, train_loader, num_epochs=10, learning_rate=0.001):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        for epoch in range(num_epochs):
            for inputs, labels in train_loader:
                optimizer.zero_grad()
                outputs = self(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
        print('Finished Training')

    def evaluate_model(self, test_loader):
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = self(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print(f'Accuracy: {100 * correct / total:.2f}%')

dev = qml.device('default.qubit', wires=4)

@qml.qnode(dev, interface='torch')
def quantum_conv_circuit(inputs, weights):
    qml.templates.AngleEmbedding(inputs, wires=range(4))
    qml.templates.StronglyEntanglingLayers(weights, wires=range(4))
    return [qml.expval(qml.PauliZ(i)) for i in range(4)]


optimizer = optim.RMSprop(model.parameters(), lr=0.001)