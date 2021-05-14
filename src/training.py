import os
import matplotlib.pyplot as plt
import numpy as np
import pickle

from sklearn import metrics
from sklearn.model_selection import train_test_split

from tensorflow.keras import datasets, layers, models
from keras_preprocessing.image import ImageDataGenerator


def train(data_path, model_path, width=80, height=None):

    TRAINING_DIR = data_path
    training_datagen = ImageDataGenerator(rescale = 1./255)

    train_generator = training_datagen.flow_from_directory(
        TRAINING_DIR,
        target_size=(width, height),
        color_mode="grayscale",
        class_mode='categorical',
        batch_size=126
    )
    
    shape = train_generator.image_shape
    labels = list(train_generator.class_indices.keys())

    print('Number of samples: ', train_generator.samples)
    print('Image shape: ', shape)
    print('Labels:', labels)

    cnn = models.Sequential([
        layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', input_shape=shape),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.5),
        
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(4, activation='sigmoid')
    ])

    cnn.summary()

    cnn.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
    
    cnn.fit(train_generator, epochs=30)

    cnn.save(model_path)

    pickle_out = open(os.path.join(model_path, "labels"), "wb")
    pickle.dump(labels, pickle_out)
    pickle_out.close()