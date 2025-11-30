"""
Train a CNN model for digit recognition using the MNIST dataset.

This script trains a convolutional neural network to recognize handwritten digits
(0-9) which is then used by the Sudoku solver to extract digits from puzzle images.

Usage:
    python train_digit_model.py

The trained model will be saved as 'digit_model.keras' in the current directory.
"""

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist

# Load MNIST digits
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize to [0, 1]
x_train = x_train.astype("float32") / 255.0
x_test  = x_test.astype("float32") / 255.0

# Add channel dimension
x_train = x_train[..., None]
x_test  = x_test[..., None]

num_classes = 10

model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation="relu"),
    layers.Dense(num_classes, activation="softmax"),
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.fit(x_train, y_train, epochs=3, batch_size=128, validation_split=0.1)

test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print("Test accuracy:", test_acc)

# Save the model for the Sudoku script
model.save("digit_model.keras")
print("Saved model to digit_model.keras")
