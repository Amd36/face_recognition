import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

# Load HAAR face classifier
face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Define a function to preprocess the image
def preprocess_image(img, image_size):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, (image_size, image_size))
    img_normalized = img_resized / 255.0
    img_expanded = np.expand_dims(img_normalized, axis=0)
    return img_expanded

def face_extractor(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    
    if len(faces) == 0:
        return None

    for (x, y, w, h) in faces:
        cropped_face = img[y:y+h, x:x+w]
        return cropped_face

def run_face_recognition(model, labels, image_size):
    # Initialize webcam
    frame = cv2.VideoCapture(0)

    while True:
        ret, img = frame.read()

        if not ret:
            print("Failed to grab frame")
            break

        face = face_extractor(img)

        if face is not None:
            processed_face = preprocess_image(face, image_size)
            prediction = model.predict(processed_face)
            
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

    frame.release()
    cv2.destroyAllWindows()

def predict_image(image_path, model, labels, image_size):
    img = cv2.imread(image_path)
    
    if img is None:
        print("Error: Could not read image from path:", image_path)
        return

    face = face_extractor(img)
    
    if face is None:
        print("No face detected in the image.")
        return
    
    processed_face = preprocess_image(face, image_size)
    prediction = model.predict(processed_face)

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
    # Load the class indices from the file
    with open("labels.txt", 'r') as f:
        labels = [line.strip() for line in f.readlines()]

    # Load the saved model
    model_dir = r"trained_model.keras"
    model = tf.keras.models.load_model(model_dir)

    # Set image size (should match the size used during model training)
    IMAGE_SIZE = 224

    # Predict from dataset
    image_path = r"datasets\test"

    if Path(image_path).suffix:
        predict_image(image_path, labels, IMAGE_SIZE)
    else:
        images = [str(file) for file in Path(image_path).glob('*') if file.suffix.lower() in ['.jpg', '.jpeg', '.png']]

        for image in images:
            predict_image(image, model, labels, IMAGE_SIZE)

    # Predict from webcam in realtime
    # run_face_recognition(model, labels, IMAGE_SIZE)
