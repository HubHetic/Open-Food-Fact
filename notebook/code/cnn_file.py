# =======================================
# IMPORT
# =======================================
import numpy as np
from tensorflow.keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras.applications import MobileNet
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.vgg16 import preprocess_input as prepro_vg116
from tensorflow.keras.applications.efficientnet import preprocess_input as prepro_efficient_net
from keras.applications.mobilenet import preprocess_input as prepro_mobile_net
from file_variable import PATH_DATA_CNN_TRAIN
from db_vector import function_last_layers, image_vector_CNN
# =======================================
# DEPENDANCE
# =======================================


class CNN():
    def __init__(self):
        self.MODEL = ''
        self.name_model = ('VGG16', "mobile_net", "efficient_net",
                           "efficient_netB3")
        self.fonction_preprocessing = ''

# =======================================
# FONCTION DE CLASSE
# =======================================

    def changer_format(self, path_image, size):
        """ Change the image format in path_image and preprocess it for the CNN.

        Args:
            path_image (str): access path to the image
            format (tuple(int,int)): image format

        Returns:
            [np array]: numpy array of the preprocessed image

        Exemple:
            # preproces image au format (224, 224)
            >>> path_image = "./data/nutella.jpg"
            >>> img_prep = changer_format(path_image, (224, 224))
            >>> img_prep.shape
                (1, 224, 224, 3)
        """
        img = image.load_img(path_image, color_mode='rgb', target_size=size)
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img_prep = self.fonction_preprocessing(img)
        return img_prep

    def image_to_vector(self, path_image, size):
        """ Converts the image parameters into a vector.

        Args:
            image (Image Object): path of the image we want to convert

        Returns:
            numpy Array: vector corresponding to the last layer of the CNN
        """

        image = self.changer_format(path_image, size)
        func = function_last_layers(self.MODEL)
        vector = image_vector_CNN(func, image)
        return vector

    def load_model(self, name_model=''):
        """Loads the model indicated by name_model.

        Args:
            name_model (str, optional): name of the model we want to load.
            Defaults to ''.
        """
        if name_model != '':
            if name_model in self.name_model:
                if name_model == 'VGG16':
                    self.MODEL = VGG16(weights='imagenet')
                    self.fonction_preprocessing = prepro_vg116
                elif name_model == "mobile_net":
                    self.MODEL = MobileNet()
                    self.fonction_preprocessing = prepro_mobile_net
                elif name_model == "efficient_net":
                    self.MODEL = EfficientNetB0()
                    self.fonction_preprocessing = prepro_efficient_net
                elif name_model == "efficient_netB3":
                    self.MODEL = EfficientNetB3()
                    self.fonction_preprocessing = prepro_efficient_net
            else:
                raise(Exception(f"name_model not found in {self.name_model}"))
        else:
            self.MODEL = VGG16(weights='imagenet')
            self.fonction_preprocessing = prepro_efficient_net

    def find_image(self, code_image):
        """ Return the path of the image which name is code_image.

        Args:
            code_image (string): image_id

        Returns:
            string: path of the image stored in data/image
        """
        path_image = PATH_DATA_CNN_TRAIN + code_image + '.jpg'
        return path_image

    def changer_format_image_folder(self, list_image, format):
        """ Preprocessed images in list_image with format.

        Args:
            list_image (list(string)): list of images names
            format ((int, int)): final format of the preprocessed image

        Returns:
            (numpy.array, list(string)): list of preprocessed images
            associated with their names
        """
        liste_img_prep = []
        for img in list_image:
            path = PATH_DATA_CNN_TRAIN + img
            try:
                liste_img_prep.append(self.changer_format(path, format))
            except:
                continue
        return liste_img_prep, list_image
