from src.train import train_model
from src.predictor import predict_digit

if __name__ == "__main__":
    # Train the model
    print("Training the model...")
    train_model()

    # Test prediction with a sample image
    image_path = "dataset/test_images/sample_digit.png"
    predicted_digit = predict_digit(image_path)
    print(f"Predicted digit: {predicted_digit}")
