import joblib
from skimage.transform import resize
from skimage.io import imread
from cv2 import ORB_create as sift
from prettytable import PrettyTable
import numpy as np

import cv2
from scipy.cluster.vq import vq


def load_image(path, width, height):
    im = imread(path)
    im = resize(im, (width,height))

    nx, ny, nz = im.shape
    im = im.reshape((1,nx*ny*nz))

    return im


def print_probabilities(labels, probabilities):
    table = PrettyTable(['Class', 'Probability'])
    for i in range(len(labels)):
        table.add_row([labels[i], probabilities[0][i]])

    print(table)


def predict(model_path, img_path, width, height):
    # Load the classifier, class names, scaler, number of clusters and vocabulary 
    #from stored pickle file (generated during training)
    clf, classes_names, stdSlr, k, voc = joblib.load(model_path)
    feature_extractor = sift(30)
    
    im = imread(img_path)
    _, des = feature_extractor.detectAndCompute(im, None)

    # Stack all the descriptors vertically in a numpy array
    descriptors = des[0]
    for descriptor in des[1:]:
        descriptors = np.vstack((descriptors, descriptor))

    # Calculate the histogram of features
    #vq Assigns codes from a code book to observations.
    features = np.zeros((1, k), "float32")
    words, _ = vq(des, voc)
    for w in words:
        features[0][w] += 1

    # Scale the features
    # Standardize features by removing the mean and scaling to unit variance
    features = features.reshape(1, -1)
    features = stdSlr.transform(features)

    print_probabilities(list(classes_names), clf.predict_proba(features))


if __name__ == "__main__":
    predict("A:\\!Projects\\SemanticRecognition\\model\\bovw-sklearn", "A:\\!Downloads\\dog_pers.jpg", 80, 80)
