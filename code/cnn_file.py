# =======================================
# IMPORT
# =======================================

import numpy as np
import gzip
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras.optimizers import RMSprop
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
from keras.applications.vgg16 import VGG16

# =======================================
# DEPENDANCE
# =======================================

from db_vecteur import function_last_layers, image_vecteur_CNN
from db_image import changer_format

# =======================================
# VARIABLE GLOBAL
# =======================================

MODEL = ''

# =======================================

# =======================================


def image_to_vector(path_image):
    """converts the parameter image to a vector

    Args:
        image (Image Object):

    Returns:
        numpy Array: vector corresponding to the last layer of the CNN
    """
    format = (224, 224)
    image = changer_format(path_image, format)
    func = function_last_layers(MODEL)
    vector = image_vecteur_CNN(func, image)
    return vector


def test_performance_model(model):
    """displays the performance of the model with different metrics, using graphs

    Args:
        model (model Object): already trained model that you want to test
    """
    pass


def train_model(nb_example_train=0):
    """train model cnn with its hyperparameters

    Args:
        nb_image (int): number image for train
        hp_model (dico): dictionnaire with hyperparameters

    Returns:
        Object model: model train
    """
    model = "CNN"
    return model


def create_model(name_model, dic_param):
    """create model which no save

    Args:
        summary_model (dico): model and parameter of model

    Returns:
        Object: model for creation of vector
    """
    model = "CNN"
    return model


def save_model():
    """save in a file the name of the model and the parameters after training

    Args:
        model (Object): model object
    """
    pass


def charge_model(PATH_file_model):
    """loads the model with these parameters into a file

    Args:
        PATH_file_model (string): path where model is save

    Returns:
        Object: model object
    """
    global MODEL
    MODEL = VGG16(weights='imagenet')
    return MODEL
