import os
import matplotlib.pyplot as plt
import numpy as np
import joblib
from copy import deepcopy


from sklearn import metrics
from skimage.io import imread
from skimage.transform import resize
from sklearn.model_selection import train_test_split


# Classifiers
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC, OneClassSVM
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis


def resize_all(src, width=80, height=None):
    """
    Load images from path, resize them and write them as arrays to a dictionary, 
    together with labels and metadata.
     
    Parameter
    ---------
    src: str
        path to data
    width: int
        target width of the image in pixels
    """

    height = height if height is not None else width

    classes = list()
 
    # Read all images in PATH, resize and write to DESTINATION_PATH
    for subdir in os.listdir(src):
        data = dict()
        data['label'] = subdir
        data['data'] = list()

        current_path = os.path.join(src, subdir)

        for file in os.listdir(current_path):
            if file[-3:] in {'jpg', 'png'}:
                im = imread(os.path.join(current_path, file))
                im = resize(im, (width, height))

                if im.shape != (width, height, 3):
                    continue

                data['data'].append(im)

        classes.append(deepcopy(data))

    return classes


def reshape_image_data(array):
    nsamples, nx, ny, nz = array.shape
    new_array = array.reshape((nsamples,nx*ny*nz))

    return new_array


def load_image(path, width, height):
    im = imread(path)
    im = resize(im, (width,height))

    nx, ny, nz = im.shape
    im = im.reshape((1,nx*ny*nz))

    return im


def train(data_path, model_path, width=80, height=None):
    classes = resize_all(src=data_path, width=width, height=height)

    for data in classes:
        print('#' * 30)
        print('Class: ', data['label'])
        print('Number of samples: ', len(data['data']))
        print('Image shape: ', data['data'][0].shape)

        X = np.array(data['data'])
        y = np.array(data['label'])
    
        # X_train, X_test, _, _ = train_test_split(
        #     X, 
        #     test_size=0.2, 
        #     shuffle=True,
        #     random_state=42,
        # )

        X_train = reshape_image_data(X)
        # X_test = reshape_image_data(X_test)

        classifier = OneClassSVM(gamma='scale', nu=0.01)
        print('Chosen classifier:', classifier.__class__.__name__)

        classifier.fit(X_train)

        print(classifier.predict(load_image('D://Загрузки//6.jpg', 80, 80)))

        # y_pred = classifier.predict(X_test)
        # print(metrics.classification_report(y_test, y_pred))

        joblib.dump(classifier, os.path.join(model_path, data['label'] + '.pkl'))