# =======================================
# IMPORT
# =======================================
import numpy as np
from tensorflow.keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras.applications import MobileNet
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.vgg16 import preprocess_input as prepro_vg116
from tensorflow.keras.applications.efficientnet import preprocess_input as prepro_efficient_net
from keras.applications.mobilenet import preprocess_input as prepro_mobile_net
from file_variable import PATH_DATA_IMAGE
from db_vecteur import function_last_layers, image_vecteur_CNN
# =======================================
# DEPENDANCE
# =======================================


class CNN():
    def __init__(self):
        self.MODEL = ''
        self.name_model = ('VGG16', "mobile_net", "efficient_net")
        self.fonction_preprpocessing = ''

# =======================================
# FONCTION DE CLASSE
# =======================================

    def changer_format(self, path_image, format):
        """changer le format de l'image dans path_image et la préprocess pour le CNN

        Args:
            path_image (str): chemin d'accès de l'image
            format (tuple(int,int)): format d'image

        Returns:
            [np array]: numpy array de l'image préprocess

        Exemple:
            # preproces image au format (224, 224)
            >>> path_image = "./data/nutella.jpg"
            >>> img_prep = changer_format(path_image, (224, 224))
            >>> img_prep.shape
                (1, 224, 224, 3)
        """
        img = image.load_img(path_image, color_mode='rgb', target_size=format)
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img_prep = self.fonction_preprpocessing(img)
        return img_prep

    def image_to_vector(self, path_image):
        """converts the parameter image to a vector

        Args:
            image (Image Object):

        Returns:
            numpy Array: vector corresponding to the last layer of the CNN
        """
        format = (224, 224)
        image = self.changer_format(path_image, format)
        func = function_last_layers(self.MODEL)
        vector = image_vecteur_CNN(func, image)
        return vector

    def train_model(self, list_image_train):
        """train model cnn with its hyperparameters

        Args:
            nb_image (int): number image for train
            hp_model (dico): dictionnaire with hyperparameters

        Returns:
            Object model: model train
        """
        list_image_train
        model = "CNN"
        return model

    def charge_model(self, name_model=''):
        """loads the model with these parameters into a file

        Args:
            PATH_file_model (string): path where model is save

        Returns:
            Object: model object
        """
        if name_model != '':
            if name_model in self.name_model:
                if name_model == 'VGG16':
                    self.MODEL = VGG16(weights='imagenet')
                    self.fonction_preprpocessing = prepro_vg116
                elif name_model == "mobile_net":
                    self.MODEL = MobileNet()
                    self.fonction_preprpocessing = prepro_mobile_net
                elif name_model == "efficient_net":
                    self.MODEL = EfficientNetB0()
                    self.fonction_preprpocessing = prepro_efficient_net
            else:
                raise(Exception(f"name_model not found in {self.name_model}"))
        else:
            self.MODEL = VGG16(weights='imagenet')
            self.fonction_preprpocessing = prepro_efficient_net

    def find_image(self, code_image):
        """retourne le chemin de l'image dont le nom est code_image

        Args:
            code_image (string): id image

        Returns:
            string: chemin relatif de l'image stocké dans le dossier data/image
        """
        path_image = PATH_DATA_IMAGE + '/' + code_image + '.jpg'
        return path_image

    def changer_format_image_folder(self, list_image, format):
        """préprocess les images stocké dans list_image avec format

        Args:
            list_image (list(string)): nom des différents image
            format ((int, int)): format final de l'image preprocess

        Returns:
            (numpy.array, list(string)): le couple images préprocess et le
            nom des images
        """
        liste_img_prep = []
        FORMAT = (224, 224)
        for img in list_image:
            path = PATH_DATA_IMAGE + "/" + img
            try:
                liste_img_prep.append(self.changer_format(path, FORMAT))
            except:
                continue
        return liste_img_prep, list_image
