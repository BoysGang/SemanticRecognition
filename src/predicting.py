import joblib
import numpy as np

from skimage.transform import resize
from skimage.io import imread
from sklearn.linear_model import LogisticRegression

from tensorflow.keras import models
from tensorflow.keras.preprocessing import image

from prettytable import PrettyTable


def print_probabilities(labels, probabilities):
    table = PrettyTable(['Class', 'Probability'])
    for i in range(len(labels)):
        table.add_row([labels[i], probabilities[0][i]])

    print(table)


def predict(model_path, img_path, width, height):
    model = models.load_model(model_path)

    img = image.load_img(img_path, target_size=(width, height))
    x = image.img_to_array(img) / 255
    x = np.expand_dims(x, axis=0)

    images = np.vstack([x])
    probabilities = model.predict(images, batch_size=10)

    print(probabilities.tolist()[0])