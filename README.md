# AI Image Classifier

This repository contains an AI-based image classification project built using TensorFlow/Keras and OpenCV. The model is trained to classify images into different categories and can be used for various applications, such as object recognition and automated tagging.

## Features
- Uses Convolutional Neural Networks (CNNs) for image classification
- Preprocessing and augmentation using OpenCV
- Training and evaluation using TensorFlow/Keras
- Supports model saving and loading
- Simple command-line interface for predictions

## Installation

To get started, clone the repository and install the required dependencies:

```bash
git clone https://github.com/imskylab/ai-image-classifier.git
cd ai-image-classifier
pip install -r requirements.txt
```

## Dataset Preparation

Ensure you have a dataset of images organized into labeled folders:
```
/dataset/
    /class_1/
        image1.jpg
        image2.jpg
    /class_2/
        image1.jpg
        image2.jpg
```

You can modify the dataset path in the script accordingly.

## Training the Model

To train the model, run the following command:
```bash
python train.py --dataset ./dataset --epochs 10 --batch_size 32
```
Adjust parameters as needed.

## Making Predictions

To classify an image using the trained model:
```bash
python predict.py --image sample.jpg
```

## Model Evaluation

Evaluate the trained model on a test dataset:
```bash
python evaluate.py --test_dataset ./test_dataset
```

## Saving and Loading Models

The model is saved in the `models/` directory after training. You can load a saved model for inference:
```python
from tensorflow.keras.models import load_model
model = load_model('models/image_classifier.h5')
```

## Contributing

Contributions are welcome! Feel free to fork this repository and submit a pull request with improvements.

## License

This project is licensed under the MIT License.

## Contact
For any queries, reach out to yentraj@gmail.com.

