"""
Face Detection and Extraction Script

This script processes a dataset of images, detects faces using OpenCV's HAAR Cascade classifier, 
and extracts the face from each image. The extracted faces are resized to a fixed resolution and saved back into the dataset directory.
Images without detected faces are counted and logged. The script also removes the original images after face extraction.

### Features:
1. **Face Detection**: Detects faces using the pre-trained HAAR Cascade classifier.
2. **Face Extraction and Resizing**: Extracted faces are resized to a standard size (300x300 pixels).
3. **File Management**: Saves extracted faces and removes the original images.
4. **Error Reporting**: Logs the number of images where no face was detected.

### Input:
- A dataset directory with subdirectories representing different classes (e.g., different people or categories).
- Each subdirectory contains images in `.jpg`, `.jpeg`, or `.png` format.

### Output:
- Extracted faces are saved in the same subdirectory as the original image, named sequentially (e.g., `face_1.jpg`, `face_2.jpg`).
- Images without faces are removed from the dataset, and a count of such images is printed.

### Command-line Arguments:
- `--dataset_dir`: Path to the dataset directory. The default is `"datasets"`.

### Dependencies:
- OpenCV
- os
- pathlib
- argparse

"""

import cv2
import os
from pathlib import Path
import argparse

# Load HAAR face classifier
face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def face_extractor(img):
    """Extracts a single face from an image and resizes it to 300x300 pixels."""
    # Convert image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    
    # If one face is detected, crop and resize it
    if len(faces) == 1:
        x, y, w, h = faces[0]
        cropped_face = img[y:y+h, x:x+w]
        resized_face = cv2.resize(cropped_face, (300, 300))
        return resized_face
    else:
        return None
    
def face_loader(dir, labels):
    """Processes the dataset, detects and saves faces, and removes the original images."""
    for label in labels:
        class_dir = os.path.join(dir, label)
        i = 1
        count = 0
        
        # Iterate through all images in the class directory
        for image in os.listdir(class_dir):
            # Check for valid image file extensions
            if Path(image).suffix.lower() in ['.jpg', '.jpeg', '.png']:
                img = cv2.imread(os.path.join(class_dir, image))
                face = face_extractor(img)

                # If a face is detected, save the extracted face and remove the original image
                if face is not None:
                    base_name = 'face_'
                    face_path = os.path.join(class_dir, f"{base_name}{i}.jpg")
                    cv2.imwrite(face_path, face)
                    i += 1
                    os.remove(os.path.join(class_dir, image))
                else:
                    count += 1
        
        # Log the results for each class
        print(f"Added the faces of {label}")
        print(f"There are {count} images with no face detections!")

if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Extract faces from images in the dataset directory")
    parser.add_argument('--dataset_dir', type=str, default='datasets', help="Path to the dataset directory")
    args = parser.parse_args()

    # Load the labels from the labels.txt file
    with open('labels.txt', 'r') as file:
        labels = [label.strip() for label in file.readlines()]

    # Process the dataset to extract faces
    face_loader(args.dataset_dir, labels)

    print("Successfully extracted the faces from all images!")
