from tensorflow.keras.preprocessing import image


class ImgLoader:
    def __init__(self, img_scale, color_mode) -> None:
        self.__img_scale = img_scale
        self.__color_mode = color_mode

    def load_img(self, img_path):
        img = image.load_img(img_path, target_size=self.__img_scale, color_mode=self.__color_mode)
        img = image.img_to_array(img) / 255

        return img