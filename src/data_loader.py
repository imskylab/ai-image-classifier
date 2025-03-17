import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def load_data():
    # Load the MNIST dataset
    mnist = tf.keras.datasets.mnist
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # Normalize pixel values to range [0, 1]
    X_train, X_test = X_train / 255.0, X_test / 255.0

    # Reshape for CNN input (28x28 grayscale images)
    X_train = X_train.reshape(-1, 28, 28, 1)
    X_test = X_test.reshape(-1, 28, 28, 1)

    # Split into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

    return X_train, y_train, X_val, y_val, X_test, y_test

def show_sample_images(X_train):
    plt.figure(figsize=(5, 5))
    for i in range(9):
        plt.subplot(3, 3, i+1)
        plt.imshow(X_train[i].reshape(28, 28), cmap='gray')
        plt.axis('off')
    plt.show()
