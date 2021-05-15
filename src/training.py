import os
import numpy as np
import pickle

from sklearn.metrics import classification_report, confusion_matrix

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.keras import datasets, layers, models
from keras_preprocessing.image import ImageDataGenerator


def train(data_path, model_path, width=80, height=None):
    TRAINING_DIR = data_path
    BATCH_SIZE = 64

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
        TRAINING_DIR,
        target_size=(width, height),
        color_mode="grayscale",
        class_mode='categorical',
        batch_size=BATCH_SIZE,
        shuffle=True,
        subset="training"
    )

    validation_generator = training_datagen.flow_from_directory(
        TRAINING_DIR,
        target_size=(width, height),
        color_mode="grayscale",
        class_mode='categorical',
        batch_size=BATCH_SIZE,
        shuffle=False,
        subset="validation"
    )
    
    shape = train_generator.image_shape
    labels = list(train_generator.class_indices.keys())
    train_samples_num = train_generator.samples
    validation_samples_num = validation_generator.samples

    print('Number of samples: ', train_samples_num)
    print('Image shape: ', shape)
    print('Labels:', labels)
    print()

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
              loss='categorical_crossentropy',
              metrics=['accuracy'])
    
    print("\nTraining:")
    cnn.fit(train_generator,
            epochs=100,
            steps_per_epoch=train_samples_num // BATCH_SIZE,
            validation_data=validation_generator,
            validation_steps=validation_samples_num // BATCH_SIZE)

    # Confution Matrix and Classification Report
    Y_pred = cnn.predict(validation_generator, validation_samples_num // BATCH_SIZE+1)
    y_pred = np.argmax(Y_pred, axis=1)
    
    print('\nConfusion Matrix:')
    print(confusion_matrix(validation_generator.classes, y_pred))
    
    print('\nClassification Report:')
    print(classification_report(validation_generator.classes, y_pred, target_names=labels))

    # Save trained model
    cnn.save(model_path)

    pickle_out = open(os.path.join(model_path, "labels"), "wb")
    pickle.dump(labels, pickle_out)
    pickle_out.close()