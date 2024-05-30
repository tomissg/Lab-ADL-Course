import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import sys
from tensorflow.keras.callbacks import TensorBoard


sys.path.append('Lab_0/Functions.py')

# import custom functions from Functions.py
from Functions import simple_cnn
from Functions import learning_curves

# Create an instance of the model
model_lrsgd = simple_cnn('leaky_relu')

# Load and preprocess CIFAR-10 dataset
(x, y), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x, x_test = x / 255.0, x_test / 255.0
x_train, x_valid, y_train, y_valid = train_test_split(x ,y, test_size = 0.1, random_state=42, shuffle=True)

# One-hot encode the labels
y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
y_valid = tf.keras.utils.to_categorical(y_valid, num_classes=10)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)


#Setting learning rate at 0.0001
sgd_optimizer = tf.keras.optimizers.SGD(learning_rate=0.0001)

# Compile the model with SGD optimizer
model_lrsgd.compile(optimizer='sgd',
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Train the model
tb_callback = TensorBoard(log_dir='./logs//{}'.format('model_lrsgd'), histogram_freq=1)
history_lrsgd = model_lrsgd.fit(x_train, y_train, epochs=10, batch_size=64, validation_data=(x_valid, y_valid), callbacks = [tb_callback])

#Plotting learning curve on accuracy and loss for CNN with sgd optimiser and LeakyRelu
learning_curves(history_lrsgd)

#Creatin the model with adam optimiser
model_lradam = simple_cnn('leaky_relu')

#Setting learning rate at 0.0001
adam_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

# Compile the model with Adam optimizer
model_lradam.compile(optimizer='adam',
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Train the model
tb_callback = TensorBoard(log_dir='./logs//{}'.format('model_lradam'), histogram_freq=1)
history_lradam = model_lradam.fit(x_train, y_train, epochs=10, batch_size=64, validation_data=(x_valid, y_valid), callbacks = [tb_callback])

#Plotting learning curve on accuracy and loss for CNN with Adam optimiser and LeakyRelu
learning_curves(history_lradam)

#Creatin the model with adam optimiser
model_tanhadam = simple_cnn('tanh')

# Compile the model with Adam optimizer
model_tanhadam.compile(optimizer='adam',
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Train the model
tb_callback = TensorBoard(log_dir='./logs//{}'.format('model_tanhadam'), histogram_freq=1)
history_tanhadam = model_tanhadam.fit(x_train, y_train, epochs=10, batch_size=64, validation_data=(x_valid, y_valid), callbacks = [tb_callback])

#Plotting learning curve on accuracy and loss for CNN with Adam optimiser and Tanh
learning_curves(history_tanhadam)



