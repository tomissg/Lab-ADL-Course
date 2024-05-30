import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import sys
sys.path.append('Lab_0/Functions.py')
from Functions import download_svhn,load_svhn
from skimage.color import rgb2gray
from skimage.transform import resize
import numpy as np
from sklearn.model_selection import train_test_split


# Load and preprocess the MNIST dataset
(x, y), (x_test, y_test) = mnist.load_data()
x = x.reshape((x.shape[0], 28, 28, 1)).astype('float32') / 255
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)
x_test = x_test.reshape((x_test.shape[0], 28, 28, 1)).astype('float32') / 255
y_train = to_categorical(y_train, 10)
y_val = to_categorical(y_val, 10)
y_test = to_categorical(y_test, 10)

# Define the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=10, batch_size=128, validation_data=(x_val, y_val))

# Evaluate the model on the test data
mnist_loss, mnist_accuracy = model.evaluate(x_test, y_test)
print(f'MNIST Test accuracy: {mnist_accuracy:.4f}')

# Load and preprocess the SVHN dataset

data_dir = './data_task_3'
download_svhn(data_dir)

x_svhn, y_svhn = load_svhn(r'C:\Users\thoma\MasterProjects\Lab_0\data_task_3\train_32x32.mat')
x_test_svhn, y_test_svhn = load_svhn(r'C:\Users\thoma\MasterProjects\Lab_0\data_task_3\test_32x32.mat')

# Convert SVHN images to grayscale and resize to 28x28
x_svhn_gray = np.array([resize(rgb2gray(img), (28, 28, 1)) for img in x_svhn])
x_train_svhn_gray, x_val_svhn_gray, y_train_svhn, y_val_svhn = train_test_split(x_svhn_gray, y_svhn, test_size=0.2, random_state=42)
x_test_svhn_gray = np.array([resize(rgb2gray(img), (28, 28, 1)) for img in x_test_svhn])

x_train_svhn_gray = x_train_svhn_gray.astype('float32') / 255
x_test_svhn_gray = x_test_svhn_gray.astype('float32') / 255
y_train_svhn = to_categorical(y_train_svhn, 10)
y_val_svhn = to_categorical(y_val_svhn, 10)
y_test_svhn = to_categorical(y_test_svhn, 10)

# Use the trained MNIST model as a base model for SVHN
model_svhn = Sequential(model.layers[:-1])  # Copy all layers except the last one
model_svhn.add(Dense(10, activation='softmax'))  # Add a new output layer

# Compile the new model
model_svhn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the new model on the SVHN dataset
model_svhn.fit(x_train_svhn_gray, y_train_svhn, epochs=10, batch_size=128, validation_data=(x_val_svhn_gray, y_val_svhn))

# Evaluate the model on the SVHN test data
svhn_loss, svhn_accuracy = model_svhn.evaluate(x_test_svhn_gray, y_test_svhn)
print(f'SVHN Test accuracy: {svhn_accuracy:.4f}')

# Use the trained MNIST features as a base model for SVHN
model_svhn_fe = Sequential(model.layers[:-1])  # Copy all layers except the last one
model_svhn_fe.add(Dense(10, activation='softmax'))  # Add a new output layer

# Freeze all but the last layer for transfer learning
for layer in model_svhn_fe.layers[:-1]:
    layer.trainable = False

# Compile the model with frozen layers
model_svhn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Fine-tune the model on the SVHN dataset
model_svhn.fit(x_train_svhn_gray, y_train_svhn, epochs=10, batch_size=128, validation_data=(x_val_svhn_gray, y_val_svhn))

# Evaluate the fine-tuned model on the SVHN test data
svhn_fine_tune_loss, svhn_fine_tune_accuracy = model_svhn.evaluate(x_test_svhn_gray, y_test_svhn)
print(f'SVHN Test accuracy after fine-tuning on mnist model: {svhn_fine_tune_accuracy:.4f}')