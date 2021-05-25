import os
import numpy as np
import joblib


from sklearn.model_selection import train_test_split
from skimage.io import imread
from skimage.transform import resize

from cv2 import ORB_create as sift
from scipy.cluster.vq import kmeans, vq
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import classification_report, confusion_matrix



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

    labels = os.listdir(src)
    mlb = MultiLabelBinarizer()
    mlb.fit([labels])

    feature_extractor = sift(30)
 
    # Read all images in PATH, resize and write to DESTINATION_PATH
    for subdir in os.listdir(src):
        print(subdir)
        current_path = os.path.join(src, subdir)

        for file in os.listdir(current_path):
            if file[-3:] in {'jpg', 'png'}:
                im = imread(os.path.join(current_path, file)).astype(np.float32)
                im = resize(im, (width, height))

                if im.shape != (width, height, 3):
                    continue
                
                _, des = feature_extractor.detectAndCompute(im, None)
                
                if des is not None:
                    data['descriptors'].append(des)   
                    data['label'].append(mlb.transform([{subdir}]).reshape(len(labels)))
                    data['filename'].append(file)
                    data['data'].append(im)

    return data


def reshape_image_data(array):
    nsamples, nx, ny, nz = array.shape
    new_array = array.reshape((nsamples,nx*ny*nz))

    return new_array


def train(data_path, model_path, width=80, height=None):
    data = resize_all(src=data_path, width=width, height=height)
    
    labels = os.listdir(data_path)
    
    print('Number of samples: ', len(data['data']))
    print('Image shape: ', data['data'][0].shape)
    print('Labels:', labels)

    descriptors = data["descriptors"][0]
    for descriptor in data["descriptors"][1:]:
        descriptors = np.vstack((descriptors, descriptor))

    #kmeans works only on float, so convert integers to float
    descriptors_float = descriptors.astype(float)

    # Perform k-means clustering and vector quantization
    k = 200 
    voc, _ = kmeans(descriptors_float, k, 1)

    # Calculate the histogram of features and represent them as vector
    #vq Assigns codes from a code book to observations.
    im_features = np.zeros((len(data['data']), k), "float32")
    for i in range(len(data['data'])):
        words, _ = vq(data["descriptors"][i] ,voc)
        for w in words:
            im_features[i][w] += 1

    # Scaling the words
    # Standardize features by removing the mean and scaling to unit variance
    # In a way normalization
    stdSlr = StandardScaler().fit(im_features)
    im_features = stdSlr.transform(im_features)
    

    X_train, X_test, y_train, y_test = train_test_split(
        im_features, 
        np.array(data['label']), 
        test_size=0.2, 
        shuffle=True,
        random_state=42,
    )

    # Train an algorithm to discriminate vectors corresponding to positive and negative training images
    # Train the Linear SVM
    clf = OneVsRestClassifier(SVC(kernel='linear',probability=True, max_iter=10000), n_jobs=-1)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    y_pred = [labels[y.argmax()] for y in y_pred]
    y_test = [labels[y.argmax()] for y in y_test]

    print('\nConfusion Matrix:')
    print(confusion_matrix(y_test, y_pred))
    print('\nReport:')
    print(classification_report(y_test, y_pred))

    joblib.dump((clf, labels, stdSlr, k, voc), model_path, compress=3)

if __name__ == '__main__':
    train(r"A:\!Projects\SemanticRecognition\data", "A:\\!Projects\\SemanticRecognition\\model\\bovw-sklearn")