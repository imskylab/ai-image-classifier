import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

# Load MNIST dataset
(_, _), (X_test, y_test) = mnist.load_data()

# Select a random digit from test data
index = np.random.randint(0, len(X_test))
sample_image = X_test[index]

# Save the image
plt.imsave("dataset/test_images/sample_digit.png", sample_image, cmap='gray')
print(f"Sample image saved as 'dataset/test_images/sample_digit.png' (Label: {y_test[index]})")
