import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import models, layers, regularizers
from tensorflow.keras.callbacks import ReduceLROnPlateau
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
import matplotlib.pyplot as plt

# Applying random transformations to the images
train_datagen = ImageDataGenerator(
    rescale=1./255,  # Normalize pixel values
    rotation_range=20,  # Randomly rotate images
    width_shift_range=0.2,  # Shift horizontally
    height_shift_range=0.2,  # Shift vertically
    zoom_range=0.2,  # Random zoom
    horizontal_flip=True  # Randomly flip images
)

val_test_datagen = ImageDataGenerator(rescale=1./255)  # Normalize validation/test data without augmenting

# Load data from directories
target_dir = 'C:/Users/AliRi/Desktop/t/Uni/IMCS/split-data'

train_generator = train_datagen.flow_from_directory(
    os.path.join(target_dir, 'train'),
    target_size=(128, 128),
    batch_size=32,
    class_mode='binary'
)

validation_generator = val_test_datagen.flow_from_directory(
    os.path.join(target_dir, 'val'),
    target_size=(128, 128),
    batch_size=32,
    class_mode='binary'
)

test_generator = val_test_datagen.flow_from_directory(
    os.path.join(target_dir, 'test'),
    target_size=(128, 128),
    batch_size=32,
    class_mode='binary',
    shuffle=False
)

# Model Architecture with Regularization
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3), 
                  kernel_regularizer=regularizers.l2(0.001)),  # L2 regularization
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    layers.Dropout(0.5),  # Dropout for additional regularization
    layers.Dense(1, activation='sigmoid')  # Output for binary classification
])

# Compilation with Adam optimizer and binary cross-entropy loss
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Learning rate scheduler callback
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.001)

# Model training
history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=10,
    validation_data=validation_generator,
    validation_steps=len(validation_generator),
    callbacks=[reduce_lr]  # Reduce learning rate when validation performance plateaus
)

# Evaluate model on the test set
test_loss, test_acc = model.evaluate(test_generator, steps=len(test_generator))
print(f"Test Accuracy: {test_acc}")

# Generate predictions and calculate confusion matrix
y_true = test_generator.classes  # True labels
y_pred = (model.predict(test_generator, steps=len(test_generator)) > 0.5).astype("int32").flatten()  # Predicted labels
conf_matrix = confusion_matrix(y_true, y_pred)

# Print confusion matrix and derived metrics
print("Confusion Matrix:")
print(conf_matrix)

# Calculate and display metrics
TP = conf_matrix[1, 1]
TN = conf_matrix[0, 0]
FP = conf_matrix[0, 1]
FN = conf_matrix[1, 0]

accuracy = (TP + TN) / np.sum(conf_matrix)
precision = TP / (TP + FP) if (TP + FP) > 0 else 0
recall = TP / (TP + FN) if (TP + FN) > 0 else 0
f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
fpr = FP / (FP + TN) if (FP + TN) > 0 else 0

print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1_score:.2f}")
print(f"False Positive Rate (FPR): {fpr:.2f}")

# Plot training and validation accuracy
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Training vs. Validation Accuracy')
plt.show()

# Plot training and validation loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Training vs. Validation Loss')
plt.show()