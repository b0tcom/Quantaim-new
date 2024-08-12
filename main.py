from pickletools import optimize
import sys
import tensorflow as tf
import numpy as np
import pennylane
from PIL import Image
from mss import mss
from utils.utils import utils
from mss import cat
from keras import Sequential
from keras import Conv2D, MaxPooling2D, BatchNormalization, Dropout, Flatten, Dense  # Import necessary layers
from tensorflow import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping # Import ModelCheckpoint, ReduceLROnPlateau, and EarlyStopping
from mss import ImageDataGenerator  # Import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchvision.models import resnet18 # Import ResNet18 model


def setup_model(model_name , device):
    """
    Load a model from the model zoo and set it to the desired device.

    :param model_name: Name of the model to load.
    :param device: Device to set the model to.
    :return: Model wrapper instance.
    """
    # Load the model from the model zoo
    model_wrapper = utils.utils.load_model ( model_name , device )

    # Set the model to the desired device
    model_wrapper.set_device ( device )

    return model_wrapper

def setup_quantum_model():
    """
    Load a quantum model from the model zoo.
    
    :return: Quantum model instance.
    """
    # Load the quantum model from the model zoo
    model = setup_quantum_model()

    return model

def setup_environment():
    """
    Set up the environment for screen capture and device configuration.
    
    :return: sct (screen capture tool instance), device (CUDA or CPU device)
    """
    # Define the screen area to capture (modify these values as needed)
    mon = {'top': 0 , 'left': 0 , 'width': 500 , 'height': 500}

    # Initialize the screen capture tool (mss)
    sct = mss ()

    # Check if CUDA is available, otherwise use CPU
    device = utils.check_cuda ()

    return sct , device

def capture_screen(sct, mon):
    """
    Capture the screen area defined by 'mon' using the mss tool.
    
    :param sct: mss screen capture instance.
    :param mon: Dictionary defining the screen area to capture.
    :return: Captured frame as a NumPy array.
    """
    # Capture the screen region
    screenshot = sct.grab ( mon )

    # Convert the screenshot to an RGB image
    img = Image.frombytes ( 'RGB' , screenshot.size , screenshot.bgra , 'raw' , 'BGRX' )

    # Convert the image to a NumPy array for processing
    return np.array ( img )


def detect_and_target(model_wrapper , sct):
    """
    Continuously capture screen, detect targets, and aim at them using the provided model.
    
    :param model_wrapper: A model wrapper (e.g., YOLO) used for detecting objects.
    :param sct: mss screen capture instance.
    """
    # Define the monitor region for capturing the screen
    mon = {'top': 0 , 'left': 0 , 'width': 500 , 'height': 500}

    # Initialize the aimbot instance with desired parameters
    aimbot = utils.utils ( sensitivity=1.5 , smooth_factor=1.0 , fov=90 , activation_key='shift' )

    while True:
        # Capture the current frame
        frame = capture_screen ( sct , mon )

        # Use the model to detect objects in the frame
        results = model_wrapper.detect ( frame )

        # Extract detection results and target the closest object within the field of view (FOV)
        detections = results.pandas ().xyxy [ 0 ].to_dict ( orient="records" )
        aimbot.target_closest_in_fov ( detections )

    # Define a simple model
    def simple_model():
        model = yolov8n.pt ( pretrained=True )
        nn.Linear(64, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
        return model
        
# Define the RMSprop optimizer
def setup_optimizer(model):optim.rmsprop(model.parameters(), lr=0.001)
        
# Example usage in a training loop
def _create_data(self):
        x1 = torch.rand(self.num_points) * 4 - 2
        x2_ = torch.rand(self.num_points) - torch.randint(0, 2, [self.num_points]).float() * 2
        x2 = x2_ + torch.floor(x1) % 2
        self.data = torch.stack([x1, x2]).t() * 2

criterion = nn.CrossEntropyLoss()

# Dummy input and target
input = torch.randn(32, 64)
target = torch.randint(0, 10, (32,))

# Forward pass
def model(one, two):
    """
    Purpose: one
    """
    
# end def
output = model (input)
loss = criterion(output, target)

# Backward pass and optimization
optimize.zero_grad()
loss.backward()
optimize.step()

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

# Load dataset
df = pd.read_csv('test.csv', index_col=0)
X = df.iloc[:, :100*100].values.reshape(-1, 100, 100, 1)
y = df.iloc[:, -1].values

# One-hot encode labels
y = to_categorical(y, num_classes=1 + df['class'].nunique())

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

# Data augmentation
train_datagen = ImageDataGenerator(rescale=1./255., rotation_range=10, width_shift_range=0.25, height_shift_range=0.25, shear_range=0.1, zoom_range=0.25)
valid_datagen = ImageDataGenerator(rescale=1./255.)


# Model definition
model = Sequential(name='Face_trained_model_' + datetime.now().strftime("%H_%M_%S_"))
model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(100, 100, 1)))
model.add(BatchNormalization())
model.add(Conv2D(64, kernel_size=3, activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(64, kernel_size=5, padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(128, kernel_size=3, activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(128, kernel_size=3, activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(128, kernel_size=5, padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(256, kernel_size=3, activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(y.shape[1], activation='softmax'))

# Compile model
model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Callbacks
checkpoint = ModelCheckpoint('best_model.h5', save_best_only=True, monitor='val_loss', mode='min')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train model
history = model.fit(train_datagen.flow(X_train, y_train, batch_size=32), validation_data=valid_datagen.flow(X_test, y_test), epochs=50, callbacks=[checkpoint, reduce_lr, early_stopping])

# Evaluate model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test accuracy: {accuracy:.2f}')
