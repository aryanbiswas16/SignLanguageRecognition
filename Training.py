import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.layers import Dense, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator

# If you have more than one GPU, you may need to modify this to select the appropriate GPU or handle multiple GPUs.
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the base model layers
base_model.trainable = False

# Build the Sequential model using mobilenet v2 model
# Tried creating my own but failed to be accurate and there seemed to be many issues  
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(1024, activation='relu'),
    Dense(26, activation='softmax')  # Assuming 26 classes for the dataset
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Paths to your data directories
train_path = 'formatedDATA/train'
valid_path = 'formatedDATA/valid'
test_path = 'formatedDATA/test'

# ImageDataGenerators
train_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input).flow_from_directory(
    directory=train_path, target_size=(300, 300), batch_size=10, class_mode='categorical')

valid_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input).flow_from_directory(
    directory=valid_path, target_size=(300, 300), batch_size=10, class_mode='categorical')

test_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input).flow_from_directory(
    directory=test_path, target_size=(300, 300), batch_size=10, class_mode='categorical')

# Fit the model
model.fit(train_batches,
          steps_per_epoch=len(train_batches),
          validation_data=valid_batches,
          validation_steps=len(valid_batches),
          epochs=2,  # You can change the number of epochs
          verbose=1)

# Save the model
model.save('asl_model.h5')

# Evaluate the model
test_loss, test_accuracy = model.evaluate(test_batches, steps=len(test_batches))
print('Test accuracy:', test_accuracy)