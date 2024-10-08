"""
Face Cropper and Dataset Collector

This script uses the Picamera2 library to capture images from a Raspberry Pi camera. It detects faces in the captured frames using a HAAR face classifier and collects samples of the user's face, saving them to a specified directory. The script also updates a `labels.txt` file with the name of the dataset, which is useful for managing multiple datasets.

### Features:
- Captures face images and saves them in a specified dataset directory.
- Detects and crops faces from the captured images.
- Updates a labels file with the name of the person whose images are being captured.
- Collects up to 100 samples of the user's face.

### Command-line Arguments:
- `--name`: Name of the dataset directory (default is 'Person').
"""

import cv2
import os
import time
import argparse
from picamera2 import Picamera2

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Face Cropper and Dataset Collector")
parser.add_argument('--name', type=str, default='Person', help="Name of the dataset directory")
args = parser.parse_args()

# Load HAAR face classifier
face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Define the base dataset dir
base_dir = "datasets"

# Load functions
def face_extractor(img):
    # Function detects faces and returns the cropped face
    # If no face detected, it returns None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        return None

    # Crop the first face found
    for (x, y, w, h) in faces:
        cropped_face = img[y:y+h, x:x+w]
        return cropped_face
    
def update_labels(dataset_dir):
    labels = [f for f in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, f))]

    with open("labels.txt", 'w') as file:
        for label in labels:
            file.write(f"{label}\n")


# Initialize Picamera2
picam2 = Picamera2()
picam2.start()

# Set dataset directory based on user input
dataset_dir = os.path.join(base_dir, args.name)
if not os.path.exists(dataset_dir):
    os.makedirs(dataset_dir)

count = 0

# Collect 100 samples of your face from the Picamera2 input
try:
    while True:
        # Start timing for FPS calculation
        start_time = time.time()

        # Capture frame from Picamera2
        frame = picam2.capture_array()

        if face_extractor(frame) is not None:
            count += 1
            face = cv2.resize(face_extractor(frame), (300, 300))

            # Save file in specified directory with a unique name
            file_name_path = os.path.join(dataset_dir, "sample" + str(count) + '.jpg')
            cv2.imwrite(file_name_path, face)

            # Put count on images and display live count
            cv2.putText(face, str(count), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('Face Cropper', face)

        else:
            print("Face not found")
            pass

        # Calculate and display FPS
        elapsed_time = time.time() - start_time
        fps = 1 / elapsed_time
        print(f"FPS: {fps:.2f}")

        # Break the loop if 'Esc' key is pressed or 100 samples have been collected
        if cv2.waitKey(1) == 27 or count == 100:  # 27 is the Esc Key
            break

finally:
    # Release resources
    picam2.close()
    cv2.destroyAllWindows()
    print("Samples Taken")

    update_labels(base_dir)
    print("labels.txt updated!")
