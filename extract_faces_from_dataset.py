import cv2
import os
from pathlib import Path

# Load HAAR face classifier
face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def face_extractor(img):
    # Convert image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    
    if len(faces) == 1:
        x, y, w, h = faces[0]
        cropped_face = img[y:y+h, x:x+w]
        resized_face = cv2.resize(cropped_face, (300, 300))
        return resized_face
    else:
        return None
    
def face_loader(dir, labels):
    for label in labels:
        class_dir = os.path.join(dir, label)
        i = 1
        count = 0
        
        for image in os.listdir(class_dir):
            # Check if the image has a valid extension
            if Path(image).suffix.lower() in ['.jpg', '.jpeg', '.png']:
                img = cv2.imread(os.path.join(class_dir, image))
                face = face_extractor(img)

                if face is not None:
                    base_name = 'face_'
                    face_path = os.path.join(class_dir, f"{base_name}{i}.jpg")
                    cv2.imwrite(face_path, face)
                    i += 1
                    os.remove(os.path.join(class_dir, image))
                else:
                    count = count + 1
        
        print("Added the faces of", label)
        print(f"There are {count} images with no face detections!")

if __name__ == '__main__':
    # Change this directory to the dataset directory of your choice
    dataset_dir = "datasets2"

    with open('labels.txt', 'r') as file:
        labels = [label.strip() for label in file.readlines()]

    face_loader(dataset_dir, labels)

    print("Successfully extracted the faces from the all images!")
