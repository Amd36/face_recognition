import cv2
import numpy as np
import os

# Load HAAR face classifier
face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Load functions
def face_extractor(img):
    # Function detects faces and returns the cropped face
    # If no face detected, it returns the input image

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)

    if faces is ():
        return None

    # Crop all faces found
    for (x, y, w, h) in faces:
        cropped_face = img[y:y+h, x:x+w]
        return cropped_face

# Initialize Webcam
cap = cv2.VideoCapture(0)
count = 0

# Collect 100 samples of your face from webcam input
while True:
    ret, frame = cap.read()
    dataset_dir = "datasets\\Ammu\\"

    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)

    if face_extractor(frame) is not None:
        count += 1
        face = cv2.resize(face_extractor(frame), (300, 300))

        # Save file in specified directory with unique name
        file_name_path = dataset_dir + "sample" + str(count) + '.jpg'
        cv2.imwrite(file_name_path, face)

        # Put count on images and display live count
        cv2.putText(face, str(count), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Face Cropper', face)

    else:
        print("Face not found")
        pass

    if cv2.waitKey(1) == 27 or count == 100:  # 27 is the Esc Key
        break

cap.release()
cv2.destroyAllWindows()
print("Samples Taken")
