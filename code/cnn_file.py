# =======================================
# IMPORT
# =======================================
import os
import numpy as np
import gzip
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras.optimizers import RMSprop
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
from keras.applications.vgg16 import VGG16
from db_vecteur import function_last_layers, image_vecteur_CNN
from db_image import changer_format

# =======================================
# DEPENDANCE
# =======================================


class CNN():
    def __init__(self):
        self.MODEL = ''
        self.name_model = ('VGG16', "MNET", "ENET")

# =======================================
# FONCTION DE CLASSE
# =======================================

    def image_to_vector(self, path_image):
        """converts the parameter image to a vector

        Args:
            image (Image Object):

        Returns:
            numpy Array: vector corresponding to the last layer of the CNN
        """
        format = (224, 224)
        image = changer_format(path_image, format)
        func = function_last_layers(self.MODEL)
        vector = image_vecteur_CNN(func, image)
        return vector

    def test_performance_model(self, model):
        """displays the performance of the model with different metrics, using graphs

        Args:
            model (model Object): already trained model that you want to test
        """
        pass

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

    def save_model(self):
        """save in a file the name of the model and the parameters after training

        Args:
            model (Object): model object
        """
        pass

    def charge_model(self, PATH_file_model='', name_model=''):
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
            else:
                raise(Exception(f"name_model not found in {self.name_model}"))
        elif PATH_file_model != '':
            if os.path.exists(PATH_file_model):
                # TODO fonction a faire pour récuperer un model dans un fichier
                pass
            else:
                raise(Exception(f"path not exist:  {PATH_file_model}"))
        else:
            self.MODEL = VGG16(weights='imagenet')
