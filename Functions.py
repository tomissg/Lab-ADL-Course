import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import torch
from scipy.io import loadmat
import numpy as np
import os
import urllib.request

def simple_cnn(activation_function):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation=activation_function, input_shape=(32, 32, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation=activation_function),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation=activation_function),
        layers.Flatten(),
        layers.Dense(64, activation=activation_function),
        layers.Dense(10)
    ])
    return model

def learning_curves(history):
    # Plot training & validation accuracy values
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plot training & validation loss values
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()

def test_model(model, criterion, test_loader):
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = correct / total
    print(f'Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.4f}')

def train_model(model, optimizer, criterion, scheduler, train_loader, val_loader, epochs):
    for epoch in range(epochs):
        model.train()  # Set model to training mode
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)

        # Update learning rate
        scheduler.step()

        # Validate the model
        model.eval()  # Set model to evaluation mode
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        val_loss /= len(val_loader.dataset)
        val_accuracy = correct / total

        # Print epoch statistics)
        print(
            f'Epoch {epoch + 1}/{epochs}, Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')

# Helper function to load the SVHN dataset
def load_svhn(path, is_train=True):
    data = loadmat(path)
    images = np.transpose(data['X'], (3, 0, 1, 2))
    labels = data['y']
    labels[labels == 10] = 0  # Convert label 10 to 0 for consistency
    return images, labels


# Function to download SVHN dataset
def download_svhn(data_dir):
    url_train = 'http://ufldl.stanford.edu/housenumbers/train_32x32.mat'
    url_test = 'http://ufldl.stanford.edu/housenumbers/test_32x32.mat'

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    train_path = os.path.join(data_dir, 'train_32x32.mat')
    test_path = os.path.join(data_dir, 'test_32x32.mat')

    if not os.path.exists(train_path):
        print('Downloading train_32x32.mat...')
        urllib.request.urlretrieve(url_train, train_path)
    if not os.path.exists(test_path):
        print('Downloading test_32x32.mat...')
        urllib.request.urlretrieve(url_test, test_path)