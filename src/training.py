import os
import matplotlib.pyplot as plt
import numpy as np
import joblib


from sklearn import metrics
from skimage.io import imread
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


from tensorflow.keras import datasets, layers, models
from keras_preprocessing.image import ImageDataGenerator


def train(data_path, model_path, width=80, height=None):

    TRAINING_DIR = data_path
    training_datagen = ImageDataGenerator(rescale = 1./255)

    train_generator = training_datagen.flow_from_directory(
        TRAINING_DIR,
        target_size=(width, height),
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
        
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(2, activation='softmax')
    ])

    cnn.summary()

    cnn.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
    
    cnn.fit(train_generator, epochs=5)

    # Cannot add just list of labels
    cnn.labels = np.array(labels)
    
    cnn.save(model_path)