import joblib
from skimage.transform import resize
from skimage.io import imread
from sklearn.linear_model import LogisticRegression
from prettytable import PrettyTable


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
    model = joblib.load(model_path)
    image = load_image(img_path, width, height)

    print_probabilities(list(model.classes_), model.predict_proba(image))

if __name__ == "__main__":
    predict("A:\\!Projects\\SemanticRecognition\\model\\multilabel-sklearn", "A:\\!Downloads\\car.jpg", 80, 80)