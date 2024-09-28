"""
Face Recognition Script using a Pre-trained TFLite Model (MobileNetV2) for Real-time and Image-based Recognition

This script performs face recognition using a TensorFlow Lite (TFLite) model. 
It can work in two modes:
1. **Directory Mode**: Predicts face labels from images in a specified directory.
2. **Webcam Mode**: Performs real-time face recognition using the Raspberry Pi camera (Picamera2).

### Key Changes:
- Utilizes TensorFlow Lite (TFLite) interpreter for predictions.
- Model is expected to be in `.tflite` format with class labels stored in `labels.txt`.
- Uses HAAR Cascade Classifier for face detection.

### Command-line Arguments:
- `--source`: Specifies the input source. Options are:
  - `directory`: Predicts face labels from images in a directory.
  - `webcam`: Uses PiCamera2 for real-time face recognition.
- `--test_dir`: (Required if `--source directory` is used) Directory containing test images.

### Functions:
- `preprocess_image(img, image_size)`: Preprocesses an image (resize, normalize) for TFLite interpreter input.
- `face_extractor(img)`: Detects faces in an image and extracts the face region.
- `run_face_recognition(tflite_interpreter, labels, image_size)`: Runs real-time face recognition using PiCamera2.
- `predict_image(image_path, tflite_interpreter, labels, image_size)`: Predicts the face label from a single image using the TFLite interpreter.

### Dependencies:
- TensorFlow Lite (TFLite)
- OpenCV
- Picamera2 (for real-time recognition on Raspberry Pi)
- Matplotlib
- NumPy
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from picamera2 import Picamera2
from tflite_runtime.interpreter import Interpreter

# Load HAAR face classifier
face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Function to preprocess the image for TFLite model input
def preprocess_image(img, image_size):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, (image_size, image_size))
    img_normalized = img_resized / 255.0
    img_expanded = np.expand_dims(img_normalized, axis=0)
    return img_expanded

# Function to detect and extract faces from an image
def face_extractor(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    
    if len(faces) == 0:
        return None

    for (x, y, w, h) in faces:
        cropped_face = img[y:y+h, x:x+w]
        return cropped_face

# Function to run real-time face recognition using PiCamera2 and TFLite model
def run_face_recognition(tflite_interpreter, labels, image_size):
    # Initialize PiCamera2
    picam2 = Picamera2()
    picam2.configure(picam2.create_preview_configuration(main={"size": (640, 480)}))
    picam2.start()

    input_details = tflite_interpreter.get_input_details()
    output_details = tflite_interpreter.get_output_details()

    while True:
        img = picam2.capture_array()
        face = face_extractor(img)

        if face is not None:
            processed_face = preprocess_image(face, image_size)
            tflite_interpreter.set_tensor(input_details[0]['index'], processed_face)
            tflite_interpreter.invoke()
            prediction = tflite_interpreter.get_tensor(output_details[0]['index'])

            predicted_class = np.argmax(prediction, axis=1)[0]
            confidence = np.max(prediction) * 100
            predicted_label = labels[predicted_class]

            # Display the prediction on the frame
            cv2.putText(img, f'{predicted_label}: {confidence:.2f}%', 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.8, (255, 0, 0), 2, cv2.LINE_AA)
            
            # Draw a rectangle around the face
            x, y, w, h = face_classifier.detectMultiScale(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 1.3, 5)[0]
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

        cv2.imshow('Real-Time Face Recognition', img)

        # Break loop on 'ESC' key press
        if cv2.waitKey(1) & 0xFF == 27:  # 27 is the ESC key
            break

    picam2.stop()
    cv2.destroyAllWindows()

# Function to predict face label from an image using TFLite model
def predict_image(image_path, tflite_interpreter, labels, image_size):
    img = cv2.imread(image_path)
    
    if img is None:
        print("Error: Could not read image from path:", image_path)
        return

    face = face_extractor(img)
    
    if face is None:
        print("No face detected in the image.")
        return
    
    processed_face = preprocess_image(face, image_size)
    input_details = tflite_interpreter.get_input_details()
    output_details = tflite_interpreter.get_output_details()

    tflite_interpreter.set_tensor(input_details[0]['index'], processed_face)
    tflite_interpreter.invoke()
    prediction = tflite_interpreter.get_tensor(output_details[0]['index'])

    # Get the predicted class and confidence
    predicted_class = np.argmax(prediction, axis=1)[0]
    confidence = np.max(prediction) * 100
    predicted_label = labels[predicted_class]

    # Print the result
    print(f"Predicted label: {predicted_label} with confidence: {confidence:.2f}%")
    print(prediction)

    # Display the image with prediction
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(f"Prediction: {predicted_label} ({confidence:.2f}%)")
    plt.axis('off')
    plt.show()

if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Face recognition using a pre-trained TFLite model, either from a webcam or an image directory.")
    parser.add_argument('--source', type=str, choices=['directory', 'webcam'], required=True,
                        help="Specify 'directory' to predict from images or 'webcam' for real-time face recognition.")
    parser.add_argument('--test_dir', type=str, default=None,
                        help="Path to the directory containing test images (required if source is 'directory').")

    args = parser.parse_args()

    # Load the class indices from the file
    with open("labels.txt", 'r') as f:
        labels = [line.strip() for line in f.readlines()]

    # Load the TFLite model
    tflite_interpreter = Interpreter(model_path="trained_model.tflite")
    tflite_interpreter.allocate_tensors()

    # Set image size (should match the size used during model training)
    IMAGE_SIZE = 224

    if args.source == 'directory':
        if not args.test_dir:
            print("Error: --test_dir is required when --source is 'directory'.")
            exit(1)

        # Predict from images in the test directory
        image_path = args.test_dir

        images = [str(file) for file in Path(image_path).glob('*') if file.suffix.lower() in ['.jpg', '.jpeg', '.png']]

        for image in images:
            predict_image(image, tflite_interpreter, labels, IMAGE_SIZE)

    elif args.source == 'webcam':
        # Predict from webcam in real-time using PiCamera2
        run_face_recognition(tflite_interpreter, labels, IMAGE_SIZE)
