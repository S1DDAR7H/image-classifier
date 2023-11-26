import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator

(train_images, train_labels), (test_images, test_labels) = keras.datasets.cifar100.load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0


model = keras.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(100, activation='softmax')
])

model.compile(optimizer="adam", loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=30, verbose=2, validation_data=(test_images, test_labels))

"""test_loss, test_acc = model.evaluate(test_images, test_labels)
predictions = model.predict(test_images)
predicted_class = np.argmax(predictions[0])
true_class = test_labels[0]
print(f"Predicted class: {predicted_class}")
print(f"True class: {true_class}")
"""
model.save("image_classifier.h5")

