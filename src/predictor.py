import cv2
import numpy as np
import tensorflow as tf

def predict_digit(image_path, model_path="models/mnist_model.h5"):
    # Load the trained model
    model = tf.keras.models.load_model(model_path)

    # Load and preprocess the image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (28, 28))  # Resize to 28x28 pixels
    img = img / 255.0  # Normalize pixel values
    img = img.reshape(1, 28, 28, 1)  # Reshape for CNN input

    # Make prediction
    prediction = model.predict(img)
    predicted_digit = np.argmax(prediction)

    return predicted_digit

if __name__ == "__main__":
    image_path = "dataset/test_images/sample_digit.png"
    digit = predict_digit(image_path)
    print(f"Predicted digit: {digit}")
