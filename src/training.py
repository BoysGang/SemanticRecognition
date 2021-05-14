import os
import matplotlib.pyplot as plt
import numpy as np
import joblib


from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

import cv2
from scipy.cluster.vq import kmeans, vq
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


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
     
    data = dict()
    data['description'] = 'resized ({0}x{1}) images in rgb'.format(int(width), int(height))
    data['label'] = []
    data['filename'] = []
    data['data'] = []

    # Create feature extraction and keypoint detector objects    
    # Create List where all the descriptors will be stored
    data['descriptors'] = []

    #BRISK is a good replacement to SIFT. ORB also works but didn;t work well for this example
    orb = cv2.ORB_create(30)
 
    # Read all images in PATH, resize and write to DESTINATION_PATH
    for subdir in os.listdir(src):
        print(subdir)
        current_path = os.path.join(src, subdir)

        for file in os.listdir(current_path):
            if file[-3:] in {'jpg', 'png'}:
                grayscale = 0
                im = cv2.imread(os.path.join(current_path, file)).astype(np.float32)
                im = cv2.resize(im, (width, height))

                if im.shape != (width, height, 3):
                    continue
                
                _, des = orb.detectAndCompute(im, None)
                
                if des is not None:
                    data['descriptors'].append(des)   
                    data['label'].append(subdir)
                    data['filename'].append(file)
                    data['data'].append(im)

    return data


def reshape_image_data(array):
    nsamples, nx, ny, nz = array.shape
    new_array = array.reshape((nsamples,nx*ny*nz))

    return new_array


def train(data_path, model_path, width=80, height=None):
    data = resize_all(src=data_path, width=width, height=height)
    
    print('Number of samples: ', len(data['data']))
    print('Keys: ', list(data.keys()))
    print('Description: ', data['description'])
    print('Image shape: ', data['data'][0].shape)
    print('Labels:', np.unique(data['label']))

    descriptors = data["descriptors"][0][1]
    for descriptor in data["descriptors"][0:]:
        descriptors = np.vstack((descriptors, descriptor))

    #kmeans works only on float, so convert integers to float
    descriptors_float = descriptors.astype(float)

    # Perform k-means clustering and vector quantization
    k = 200 
    voc, variance = kmeans(descriptors_float, k, 1)

    # Calculate the histogram of features and represent them as vector
    #vq Assigns codes from a code book to observations.
    im_features = np.zeros((len(data['data']), k), "float32")
    for i in range(len(data['data'])):
        words, distance = vq(data["descriptors"][i] ,voc)
        for w in words:
            im_features[i][w] += 1

    # Perform Tf-Idf vectorization
    nbr_occurences = np.sum( (im_features > 0) * 1, axis = 0)
    idf = np.array(np.log((1.0*len(data['data'])+1) / (1.0*nbr_occurences + 1)), 'float32')

    # Scaling the words
    #Standardize features by removing the mean and scaling to unit variance
    #In a way normalization
    stdSlr = StandardScaler().fit(im_features)
    im_features = stdSlr.transform(im_features)

    #Train an algorithm to discriminate vectors corresponding to positive and negative training images
    # Train the Linear SVM
    clf = LinearSVC(max_iter=10000)  #Default of 100 is not converging
    clf.fit(im_features, np.array(data['label']))

    joblib.dump((clf, None, stdSlr, k, voc), model_path, compress=3) 