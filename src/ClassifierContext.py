import os.path
import joblib

from ImageClassifier import ImageClassifier
from ImgDataGenerator import ImgDataGenerator

class ClassifierContext:
    # Set classifier
    def set_classifier(self, classifier: ImageClassifier):
        self.__classifier = classifier

    # Train set classifier
    def fit(self, img_data_generator):
        self.__classifier.fit(img_data_generator)

    # Classify image
    def predict(self, img_path):
        return self.__classifier.predict(img_path)

    # Save model to the given path
    def save(self, path):
        self.__classifier.save(path)