import numpy as np
import joblib
import os.path
import os

import cv2
from scipy.cluster.vq import kmeans, vq, whiten

from sklearn.preprocessing import StandardScaler
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

from ImageClassifier import ImageClassifier
from ImgDataGenerator import ImgDataGenerator


class BagOfVisualWords(ImageClassifier):
    def __init__(self, max_features=30, clusters_num=200, model=None, labels=None, voc=None, scaler=None, img_loader=None):
        super().__init__()

        self.__model = model
        self._labels = labels
        self.__vocabulary = voc
        self.__clusters_num = clusters_num
        self.__extractor_max_features = max_features
        self.__feature_extractor = cv2.SIFT_create()
        self.__scaler = scaler
        self._img_loader = img_loader
    
    # Train BoVW model
    def fit(self, img_data_generator: ImgDataGenerator):
        self._init_img_loader(img_data_generator)

        train_generator = img_data_generator.train_generator
        test_generator = img_data_generator.test_generator

        self._labels = list(train_generator.class_indices.keys())

        shape = train_generator.image_shape
        train_samples_num = train_generator.samples

        print('Number of samples: ', train_samples_num)
        print('Image shape: ', shape)
        print('Labels:', self._labels)
        print()

        # Train samples
        descriptors, y_train = self.__get_data(train_generator)
        X_train, self.__vocabulary = self.__extract_features(descriptors, k=self.__clusters_num)

        # Train the Linear SVM
        self.__model = OneVsRestClassifier(SVC(kernel='linear',probability=True, max_iter=-1), n_jobs=-1)
        self.__model.fit(X_train, y_train)

        # Test samples
        test_descriptors, y_test = self.__get_data(test_generator)
        X_test, _ = self.__extract_features(test_descriptors,
                                            voc=self.__vocabulary, 
                                            k=self.__clusters_num)

        y_pred = self.__model.predict_proba(X_test)

        y_pred = [self._labels[y.argmax()] for y in y_pred]
        y_test = [self._labels[y.argmax()] for y in y_test]

        print('\nConfusion Matrix:\n')
        print(confusion_matrix(y_test, y_pred))
        
        print('\nReport:')
        print(classification_report(y_test, y_pred))

    # Classify image
    def predict(self, img_path):
        image = self._img_loader.load_img(img_path)
        
        descriptors = self.__descriptors_from_img(image)
        
        # Get descritors
        des = descriptors[0]
        for descriptor in descriptors[1:]:
            des = np.vstack((des, descriptor))

        # Calculate feature histogram
        features = np.zeros((1, self.__clusters_num), "float32")
        words, _ = vq(des, self.__vocabulary)
        for w in words:
            features[0][w] += 1

        features = features.reshape(1, -1)
        features = self.__scaler.transform(features)

        # Perform probability prediction
        probabilities = self.__model.predict_proba(features)

        return self._labels, probabilities[0]

    # Save BoVW model to the given path
    def save(self, path):
        if not os.path.exists(path):
            os.makedirs(path)

        joblib.dump((self.__model, 
                        self._labels, 
                        self.__clusters_num, 
                        self.__vocabulary,
                        self.__extractor_max_features,
                        self.__scaler,
                        self._img_loader), 
                    os.path.join(path, 'bovw.pkl'))

    # Load BoVW model from the given path
    @classmethod
    def load(cls, path):
        model, labels, clusters_num, voc, max_features, scl, loader = joblib.load(os.path.join(path, 'bovw.pkl'))
        return BagOfVisualWords(max_features, clusters_num, model, labels, voc, scl, loader)

    # Get labels and image descriptors from generator
    def __get_data(self, generator):
        samples = generator.samples
        batch_size = generator.batch_size

        descriptors, labels = list(), list()
        for _ in range(samples // batch_size + 1):
            data_batch, labels_batch = generator.next()

            for img_data, label in zip(data_batch, labels_batch):
                des = self.__descriptors_from_img(img_data)

                if des is not None:
                    descriptors.append(des)
                    labels.append(label)

        return descriptors, labels

    # Get descriptors from image
    def __descriptors_from_img(self, image_data):
        image_data *= 255
        image8bit = cv2.normalize(image_data, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
        _, des = self.__feature_extractor.detectAndCompute(image8bit, None)

        return des

    # Calculate the histogram of features
    def __extract_features(self, descriptors, voc=None, scaler=None, k=200):
        # Stack all the descriptors vertically in a numpy array
        des = np.array(descriptors[0])
        for descriptor in descriptors[1:]:
            des = np.vstack((des, descriptor))

        # Convert integers to float, so kmeans will work properly
        descriptors_float = des.astype(float)

        if voc is None:
            # Perform k-means clustering and vector quantization
            voc, _ = kmeans(whiten(descriptors_float), k, 1)

        # Calculate the histogram of features and represent them as vector
        im_features = np.zeros((len(descriptors), k), "float32")
        for i in range(len(descriptors)):
            words, _ = vq(descriptors[i], voc)
            for w in words:
                im_features[i][w] += 1

        # Standardize features by removing the mean and scaling to unit variance
        if scaler is None:
            self.__scaler = StandardScaler().fit(im_features)
        im_features = self.__scaler.transform(im_features)

        return im_features, voc

