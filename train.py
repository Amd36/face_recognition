"""
Model Training and TensorFlow Lite Conversion Script using MobileNetV2

This script trains a custom image classification model using a pre-trained MobileNetV2 as the base model.
It supports fine-tuning and converts the trained model into TensorFlow Lite (TFLite) format for deployment on resource-constrained devices such as Raspberry Pi.

### Features:
1. **Customizable Epochs**: The number of training epochs can be specified via a command-line argument.
2. **TensorFlow Lite Conversion**: The trained model is converted to TFLite format for efficient inference.
3. **Quantization**: Optional post-training quantization is applied to optimize the model further.
4. **Training History**: Saves training and validation metrics in JSON format and generates plots for visualization.

### Command-line Arguments:
- `--epochs`: Number of epochs to train the model (default is 35).

### Outputs:
- `trained_model.keras`: Saved Keras model.
- `trained_model.tflite`: Converted TFLite model.
- `trained_model_quantized.tflite`: Quantized TFLite model.
- `training_history.json`: JSON file containing the training history.
- **Accuracy and Loss Plot**: A plot showing training/validation accuracy and loss curves.

### Dependencies:
- TensorFlow
- Matplotlib
- JSON
- Argparse

"""

import tensorflow as tf
import matplotlib.pyplot as plt
import json
import argparse

# Parse command-line argument for number of epochs
parser = argparse.ArgumentParser(description='Train the model and convert to TensorFlow-Lite format')
parser.add_argument('--epochs', type=int, default=35, help='Number of epochs for training')
args = parser.parse_args()

# Define the base directory for the training images
base_dir = r"datasets"

# Define image size and batch size
IMAGE_SIZE = 224
BATCH_SIZE = 32

# Use this ImageDataGenerator without augmentation
data_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255, validation_split=0.15)

# Create training and validation data generators
train_generator = data_generator.flow_from_directory(base_dir, target_size=(IMAGE_SIZE, IMAGE_SIZE), batch_size=BATCH_SIZE, subset='training')
val_generator = data_generator.flow_from_directory(base_dir, target_size=(IMAGE_SIZE, IMAGE_SIZE), batch_size=BATCH_SIZE, subset='validation')

# Get a batch of images and labels to check the data generator
for image_batch, label_batch in train_generator:
    break

# Print class indices
print(train_generator.class_indices)

# Save the class labels to a text file
labels = '\n'.join(sorted(train_generator.class_indices.keys()))
with open("labels.txt", 'w') as f:
    f.write(labels)

# Define number of classes
nc = len(train_generator.class_indices.keys())

# Define the input shape for the model
IMAGE_SHAPE = (IMAGE_SIZE, IMAGE_SIZE, 3)

# Load the pre-trained MobileNetV2 model without the top layer
base_model = tf.keras.applications.MobileNetV2(input_shape=IMAGE_SHAPE, include_top=False, weights="imagenet")

# Unfreeze the base model to allow fine-tuning
base_model.trainable = True

# Freeze the layers up to the specified layer index to keep their weights unchanged
fine_tune_at = 100
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

# Create a new model by adding custom layers on top of the base model
model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.Conv2D(64, 3, activation='relu'),
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(1024, activation='relu'),
    tf.keras.layers.Dense(1024, activation='relu'),
    tf.keras.layers.Dense(nc, activation='softmax')    # Output layer with softmax activation for multi-class classification
])

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
epochs = args.epochs
history = model.fit(train_generator, epochs=epochs, validation_data=val_generator)

# Save the model as .keras
model.save("trained_model.keras")
print("Trained model saved as trained_model.keras")

# Convert the trained model to TensorFlow Lite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the converted model as a .tflite file
with open("trained_model.tflite", "wb") as f:
    f.write(tflite_model)

print("Model has been converted to TensorFlow Lite format and saved as trained_model.tflite")

# Optional: Apply optimizations (quantization) for Raspberry Pi
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_quantized_model = converter.convert()

# Save the quantized model as .tflite file
with open("trained_model_quantized.tflite", "wb") as f:
    f.write(tflite_quantized_model)

print("Quantized model saved as trained_model_quantized.tflite")

# Save the history as a JSON file
with open('training_history.json', 'w') as f:
    json.dump(history.history, f)

print("Created a JSON file with the training history")

# Plot the training and validation accuracy and loss
acc = history.history["accuracy"]
val_acc = history.history["val_accuracy"]
loss = history.history["loss"]
val_loss = history.history["val_loss"]

plt.figure(figsize=(12, 7))
plt.subplot(1, 2, 1)
plt.plot(acc, label="Training Accuracy")
plt.plot(val_acc, label="Validation Accuracy")
plt.legend(loc="lower right")
plt.ylabel("Accuracy")
plt.ylim([min(plt.ylim()), 1])
plt.title("Training and Validation Accuracy")

plt.subplot(1, 2, 2)
plt.plot(loss, label="Training Loss")
plt.plot(val_loss, label="Validation Loss")
plt.legend(loc="upper right")
plt.ylabel("Cross Entropy")
plt.ylim([0, 100])
plt.title("Training and Validation Loss")
plt.xlabel("Epoch")

plt.show()
