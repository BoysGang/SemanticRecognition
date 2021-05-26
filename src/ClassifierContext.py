import os.path
import joblib

from ImageClassifier import ImageClassifier
from ImgDataGenerator import ImgDataGenerator

class ClassifierContext:
    def setClassifier(self, classifier: ImageClassifier):
        self.__classifier = classifier

    def fit(self, img_data_generator):
        self.__classifier.fit(img_data_generator)

    def predict(self, img_path):
        return self.__classifier.predict(img_path)

    def save(self, path):
        self.__classifier.save(path)