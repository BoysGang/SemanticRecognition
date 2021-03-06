import os
import joblib
import numpy as np
import matplotlib.pyplot as plt

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras import regularizers

from ImageClassifier import ImageClassifier
from ImgDataGenerator import ImgDataGenerator

class ConvolutionalNeuralNetwork(ImageClassifier):
    def __init__(self, epochs=1, plot_fit_hist=True, acceleration=False, model=None, labels=None, img_loader=None):
        super().__init__()

        self.__epochs = epochs
        self.__plot_fit_hist = plot_fit_hist
        self.__acceleration = acceleration

        self.__model = model
        self._labels = labels
        self._img_loader = img_loader
        
        physical_devices = tf.config.list_physical_devices('GPU') 

        if physical_devices and acceleration:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)

    # Train CNN model
    def fit(self, img_data_generator: ImgDataGenerator):
        # Divide images into train, validation and test samples
        self._init_img_loader(img_data_generator)

        train_generator = img_data_generator.train_generator
        validation_generator = img_data_generator.validation_generator
        test_generator = img_data_generator.test_generator

        batch_size = img_data_generator.batch_size

        self._labels = list(train_generator.class_indices.keys())
        
        shape = train_generator.image_shape
        train_samples_num = train_generator.samples
        validation_samples_num = validation_generator.samples
        test_samples_num = test_generator.samples

        print('Batch size:', batch_size)
        print('Number of samples:', train_samples_num)
        print('Image shape:', shape)
        print('Labels:', self._labels)
        print()

        output_neurons = len(self._labels)

        # CNN model representation
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
            layers.Dropout(0.2),
            layers.Dense(output_neurons, activation='sigmoid')
        ])

        self.__model.summary()

        self.__model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])
        
        print("\nTraining:")
        history = self.__model.fit(train_generator,
                epochs=self.__epochs,
                steps_per_epoch=train_samples_num // batch_size,
                validation_data=validation_generator,
                validation_steps=validation_samples_num // batch_size)

        # Test model
        Y_pred = self.__model.predict(test_generator, test_samples_num // batch_size+1)
        y_pred = np.argmax(Y_pred, axis=1)
        
        print('\nConfusion Matrix:\n')
        print(confusion_matrix(test_generator.classes, y_pred))
        
        print('\nClassification Report:')
        print(classification_report(test_generator.classes, y_pred, target_names=self._labels))

        if self.__plot_fit_hist:
            self.__plot_hist(history)

    # Classify image
    def predict(self, img_path):
        image = self._img_loader.load_img(img_path)

        image = np.expand_dims(image, axis=0)
        image = np.vstack([image])

        probabilities = self.__model.predict(image, batch_size=10)

        return self._labels, probabilities[0]

    # Save CNN model to the given path
    def save(self, path):
        self.__model.save(path)

        joblib.dump(self._img_loader, os.path.join(path, "img_loader"))
        joblib.dump(self._labels, os.path.join(path, "labels"))

    # Load CNN model from the given path
    @classmethod
    def load(cls, path):
        model = models.load_model(path)
        labels =  joblib.load(os.path.join(path, "labels"))
        img_loader = joblib.load(os.path.join(path, "img_loader"))

        return ConvolutionalNeuralNetwork(model=model, labels=labels, img_loader=img_loader)

    # Plot training history
    def __plot_hist(self, history):
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']

        loss = history.history['loss']
        val_loss = history.history['val_loss']

        epochs_range = range(self.__epochs)

        plt.figure(figsize=(8,8))
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, acc, label='Train accuracy')
        plt.plot(epochs_range, val_acc, label='Validation accuracy')
        plt.legend(loc='lower right')
        plt.title('Accuracy')

        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, loss, label='Train loss')
        plt.plot(epochs_range, val_loss, label='Validation loss')
        plt.legend(loc='upper right')
        plt.title('Loss')
        plt.show()
