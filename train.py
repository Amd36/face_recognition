import tensorflow as tf
import matplotlib.pyplot as plt
import json

# Define the base directory for the training images
base_dir = r"datasets\train"

# Define image size and batch size
IMAGE_SIZE = 224
BATCH_SIZE = 32

# Create an ImageDataGenerator for data augmentation and preprocessing
# data_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255,
#                                                                  rotation_range=45,
#                                                                  width_shift_range=0.3,
#                                                                  height_shift_range=0.3,
#                                                                  horizontal_flip=True,
#                                                                  fill_mode='nearest',
#                                                                  validation_split = 0.2)

# Use this ImageDataGenerator if you want to use without augmentation
data_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255, validation_split = 0.15)

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

# Print the number of layers in the base model
print("Number of layers in the base model: ", len(base_model.layers))

# Freeze the layers up to the specified layer index to keep their weights unchanged
fine_tune_at = 100
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

# Create a new model by adding custom layers on top of the base model
model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.Conv2D(64, 3, activation='relu'),  # Convolutional layer
    # tf.keras.layers.Dropout(0.4),                     # Dropout layer for regularization
    tf.keras.layers.GlobalAveragePooling2D(),         # Global average pooling layer
    tf.keras.layers.Dense(1024, activation='relu'),     # Dense layer 1
    tf.keras.layers.Dense(1024, activation='relu'),     # Dense layer 2
    tf.keras.layers.Dense(nc, activation='softmax')    # Output layer with softmax activation for multi-class classification
])

# Compile the model with an Adam optimizer, categorical cross-entropy loss, and accuracy metric
model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Print the model summary to see the structure and number of parameters
model.summary()

# Print the number of trainable variables
print("Number of Trainable variables: ", len(model.trainable_variables))

# Define the number of epochs for training
epochs = 35

# Train the model using the training data generator and validate using the validation data generator
history = model.fit(train_generator, epochs=epochs, validation_data=val_generator)

# Get the training and validation accuracy and loss from the history object
acc = history.history["accuracy"]
val_acc = history.history["val_accuracy"]
loss = history.history["loss"]
val_loss = history.history["val_loss"]

# Plot training and validation accuracy
plt.figure(figsize=(12,7))
plt.subplot(1,2,1)
plt.plot(acc, label="Training Accuracy")
plt.plot(val_acc, label="Validation Accuracy")
plt.legend(loc="lower right")
plt.ylabel("Accuracy")
plt.ylim([min(plt.ylim()), 1])
plt.title("Training and Validation Accuracy")

# Plot training and validation loss
plt.subplot(1,2,2)
plt.plot(loss, label="Training Loss")
plt.plot(val_loss, label="Validation Loss")
plt.legend(loc="upper right")
plt.ylabel("Cross Entropy")
plt.ylim([0, 100])
plt.title("Training and Validation Loss")
plt.xlabel("Epoch")

# Show the plots
plt.show()

# Save the history as a JSON file
with open('training_history.json', 'w') as f:
    json.dump(history.history, f)

print("Created a json file with the training history")

# Save the model as .keras
model.save("trained_model.keras")
print("Trained model saved as trained_model.keras")