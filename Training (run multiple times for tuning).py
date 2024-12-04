import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator #utility used to augment image data during training
from tensorflow.keras import models, layers # models provides classes and functions to define and manage neural networks. Layers - Layers of CNN
import matplotlib.pyplot as plt #visual plotting for end

#applying random transformations to the images
train_datagen = ImageDataGenerator(
    rescale=1./255,  # Normalize pixel values
    rotation_range=20,  # Randomly rotate images
    width_shift_range=0.2,  # Shift horizontally
    height_shift_range=0.2,  # Shift vertically
    zoom_range=0.2,  # Random zoom
    horizontal_flip=True  # Randomly flip images
)

val_test_datagen = ImageDataGenerator(rescale=1./255) #Not augmenting val/test data 

# Load data from directories
target_dir = 'C:/Users/AliRi/Desktop/t/Uni/IMCS/split-data'

#automatically loads images from subdirectories
train_generator = train_datagen.flow_from_directory( 
    os.path.join(target_dir, 'train'),
    target_size=(128, 128),
    batch_size=32,
    class_mode='binary' #performing binary classification
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


model = models.Sequential([  #linear stack of layers for the network
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)), #convolutional filters to extract features from the images (32 -> 64 -> 128 filters) (3,3 -> size of window) (relu -> activation, allowing to learn complex patterns)
    layers.MaxPooling2D((2, 2)), #taking max value in a smaller window, reducing size
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(), #2D -> 1D vector
    layers.Dense(128, activation='relu'), #Fully connected - connect every input neuron to output neuron
    layers.Dense(1, activation='sigmoid')  # Output is 1 or 0, for butterfly or spider
])




#Compilation
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Step 4: Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=10,
    validation_data=validation_generator,
    validation_steps=len(validation_generator)
)

# Step 5: Evaluate the model on the test data
test_loss, test_acc = model.evaluate(test_generator, steps=len(test_generator))
print(f"Test Accuracy: {test_acc}")

# Step 6: Plot training and validation accuracy
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.show()

# Plot training and validation loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.show()