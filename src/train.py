from src.data_loader import load_data, show_sample_images
from src.model_builder import build_model
import tensorflow as tf
import os

# Function to train and save the model
def train_model():
    # Load dataset
    X_train, y_train, X_val, y_val, X_test, y_test = load_data()

    # Show sample images
    show_sample_images(X_train)

    # Build the model
    model = build_model()

    # Train the model
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=5)

    # Evaluate on test data
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {test_acc:.4f}")

    # Save model
    model_path = "models/mnist_model.h5"
    os.makedirs("models", exist_ok=True)
    model.save(model_path)
    print(f"Model saved at: {model_path}")

if __name__ == "__main__":
    train_model()
