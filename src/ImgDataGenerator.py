import joblib
import os.path

from keras_preprocessing.image import ImageDataGenerator
from scipy.sparse import data
from tensorflow.keras.preprocessing import image


class ImgDataGenerator:
    def __init__(self,
                 train_path,
                 test_path,
                 resize_to=(80, 80),
                 color_mode='grayscale',
                 validation_split=0.2,
                 rotation_range=0,
                 shear_range=0,
                 horizontal_flip=False,
                 batch_size=64):

        self.__train_path = train_path
        self.__test_path = test_path
        self.__image_scale = resize_to
        self.__color_mode = color_mode
        self.__validation_split = validation_split
        self.__rotation_range = rotation_range
        self.__shear_range = shear_range
        self.__horizontal_flip = horizontal_flip
        self.__batch_size = batch_size

        self.__train_data_generator = ImageDataGenerator(
            rescale=1./255,
            rotation_range=rotation_range,
            shear_range=shear_range,
            horizontal_flip=horizontal_flip,
            validation_split=validation_split,
            fill_mode='nearest'
        )

        self.__test_data_generator = ImageDataGenerator(rescale=1./255)

        self.__train_generator = self.__train_data_generator.flow_from_directory(
            train_path,
            target_size=resize_to,
            class_mode='categorical',
            batch_size=batch_size,
            color_mode=color_mode,
            shuffle=True,
            subset="training"
        )

        self.__validation_generator = self.__train_data_generator.flow_from_directory(
            train_path,
            target_size=resize_to,
            class_mode='categorical',
            batch_size=batch_size,
            color_mode=color_mode,
            shuffle=True,
            subset="validation"
        )

        self.__test_generator = self.__test_data_generator.flow_from_directory(
            test_path,
            target_size=resize_to,
            class_mode='categorical',
            batch_size=batch_size,
            color_mode=color_mode,
            shuffle=False
        )

    @property
    def train_generator(self):
        return self.__train_generator

    @property
    def validation_generator(self):
        return self.__validation_generator

    @property
    def test_generator(self):
        return self.__test_generator

    @property
    def batch_size(self):
        return self.__batch_size
    
    @property
    def image_scale(self):
        return self.__image_scale

    @property
    def color_mode(self):
        return self.__color_mode
