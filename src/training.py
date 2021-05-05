import os
import matplotlib.pyplot as plt
import numpy as np
import joblib


from sklearn import metrics
from skimage.io import imread
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


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
 
    # Read all images in PATH, resize and write to DESTINATION_PATH
    for subdir in os.listdir(src):
        print(subdir)
        current_path = os.path.join(src, subdir)

        for file in os.listdir(current_path):
            if file[-3:] in {'jpg', 'png'}:
                im = imread(os.path.join(current_path, file))
                im = resize(im, (width, height))

                if im.shape != (width, height, 3):
                    continue

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

    X = np.array(data['data'])
    y = np.array(data['label'])
 
    X_train, X_test, y_train, y_test = train_test_split(
        X, 
        y, 
        test_size=0.2, 
        shuffle=True,
        random_state=42,
    )

    X_train = reshape_image_data(X_train)
    X_test = reshape_image_data(X_test)

    reg_log = LogisticRegression()
    reg_log.fit(X_train, y_train)

    joblib.dump(reg_log, model_path)