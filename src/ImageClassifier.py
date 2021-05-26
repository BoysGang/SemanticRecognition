from abc import ABC, abstractmethod
from ImgDataGenerator import ImgDataGenerator
from ImgLoader import ImgLoader

class ImageClassifier(ABC):
    def __init__(self):
        self._img_loader = None

    @abstractmethod
    def fit(self, img_data_generator: ImgDataGenerator):
        pass

    @abstractmethod
    def predict(self, img_path) -> list:
        pass

    @abstractmethod
    def save(self, path):
        pass

    @abstractmethod
    def load(self, path):
        pass

    def _init_img_loader(self, img_data_generator: ImgDataGenerator):
        img_scale = img_data_generator.image_scale
        color_mode = img_data_generator.color_mode

        self._img_loader = ImgLoader(img_scale, color_mode)
