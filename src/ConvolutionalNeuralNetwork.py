import os
import joblib
import numpy as np
from prettytable import PrettyTable

from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing import image
from keras_preprocessing.image import ImageDataGenerator


class ConvolutionalNeuralNetwork:
    def __init__(self):
        self.__model = None
        self.labels = None

        physical_devices = tf.config.list_physical_devices('GPU') 

        if physical_devices:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)

    def train(self, data_path, width=80, height=80, epochs=100, batch_size=32):
        training_datagen = ImageDataGenerator(
            rescale=1./255,
            validation_split=0.2,
            rotation_range=40,
            width_shift_range=0,
            height_shift_range=0,
            shear_range=0.2,
            zoom_range=0,
            horizontal_flip=True,
            fill_mode='nearest'
        )

        train_generator = training_datagen.flow_from_directory(
            data_path,
            target_size=(width, height),
            color_mode="grayscale",
            class_mode='categorical',
            batch_size=batch_size,
            shuffle=True,
            subset="training"
        )

        validation_generator = training_datagen.flow_from_directory(
            data_path,
            target_size=(width, height),
            color_mode="grayscale",
            class_mode='categorical',
            batch_size=batch_size,
            shuffle=False,
            subset="validation"
        )
        
        self.labels = list(train_generator.class_indices.keys())
        
        shape = train_generator.image_shape
        train_samples_num = train_generator.samples
        validation_samples_num = validation_generator.samples

        print('Number of samples: ', train_samples_num)
        print('Image shape: ', shape)
        print('Labels:', self.labels)
        print()

        output_neurons = len(self.labels)

        self.__model = models.Sequential([
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
            layers.Dense(output_neurons, activation='sigmoid')
        ])

        self.__model.summary()

        self.__model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])
        
        print("\nTraining:")
        self.__model.fit(train_generator,
                epochs=epochs,
                steps_per_epoch=train_samples_num // batch_size,
                validation_data=validation_generator,
                validation_steps=validation_samples_num // batch_size)

        # Confution Matrix and Classification Report
        Y_pred = self.__model.predict(validation_generator, validation_samples_num // batch_size+1)
        y_pred = np.argmax(Y_pred, axis=1)
        
        print('\nConfusion Matrix:')
        print(confusion_matrix(validation_generator.classes, y_pred))
        
        print('\nClassification Report:')
        print(classification_report(validation_generator.classes, y_pred, target_names=self.labels))

    def predict(self, img_path, width, height):
        probabilities = predict_proba(img_path, width, height)

        return self.labels(np.argmax(probabilities))

    def predict_proba(self, img_path, width, height):
        img = image.load_img(img_path, target_size=(width, height), color_mode="grayscale")
        x = image.img_to_array(img) / 255
        x = np.expand_dims(x, axis=0)

        images = np.vstack([x])
        probabilities = self.__model.predict(images, batch_size=10)

        return probabilities

    def print_prediction(self, img_path, width, height):
        probabilities = self.predict_proba(img_path, width, height)

        table = PrettyTable(['Class', 'Probability'])
        for i in range(len(self.labels)):
            table.add_row([self.labels[i], probabilities[0][i]])

        print(table)

    def save(self, model_path):
        self.__model.save(model_path)
        joblib.dump(self.labels, os.path.join(model_path, "labels"))
        
    def load(self, model_path):
        self.__model = models.load_model(model_path)
        self.labels =  joblib.load(os.path.join(model_path, "labels"))