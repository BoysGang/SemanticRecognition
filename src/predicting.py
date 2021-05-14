import joblib
from skimage.transform import resize
from skimage.io import imread
from sklearn.linear_model import LogisticRegression
from prettytable import PrettyTable
import numpy as np

import cv2
from scipy.cluster.vq import kmeans, vq
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC   


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
    orb = cv2.ORB_create(30)
    
    im = cv2.imread(img_path)
    kpts, des = orb.detectAndCompute(im, None)

    # Stack all the descriptors vertically in a numpy array
    descriptors = des[0]
    for descriptor in des[1:]:
        descriptors = np.vstack((descriptors, descriptor))

    # Calculate the histogram of features
    #vq Assigns codes from a code book to observations.
    feature = np.zeros((k), "float32")
    words, distance = vq(des, voc)
    for w in words:
        feature[w] += 1

    # Perform Tf-Idf vectorization
    nbr_occurences = np.sum( (feature > 0) * 1, axis = 0)
    idf = np.array(np.log((2) / (1.0*nbr_occurences + 1)), 'float32')

    # Scale the features
    #Standardize features by removing the mean and scaling to unit variance
    #Scaler (stdSlr comes from the pickled file we imported)
    feature = feature.reshape(1, -1)
    feature = stdSlr.transform(feature)

    print(clf.predict(feature))
    print(clf.decision_function(feature))
