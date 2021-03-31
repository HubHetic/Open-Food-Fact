from cnn_file import image_to_vector
from knn_file import find_similar_vector_id
from db_vecteur import find_code
from db_image import find_image
from db_produit import find_line
import pandas as pd
import os
from file_variable import *
import shutil

def image_to_code(image):
    """returns the id of a similar image in the variable image

    Args:
        image ([image object]): picture jpg, gz

    Returns:
        int: id code of the image in the databases
    """
    vec = image_to_vector(image)
    vec_sim = find_similar_vector_id(vec)
    code = find_code(vec_sim)
    return find_image(code)


def image_to_line_data(image):
    """returns the product line of the image variable in the product database

    Args:
        image (Image Object): image.jpg gz

    Returns:
        pandas.Series: line in the product database
    """
    code = image_to_code(image)
    line = find_line(code)
    return line


def all_implement(path_image):
    """setting up all the folders, files, database for the production,
        training of CNN, KNN

        Args: 
        path_image (str): chemin absolu vers le dossier des images que l'on veut rentrer 
        dans le modèle
    """
    print('coucou0')

    if os.path.exists(path_image):
        print('coucou1')
        if not os.path.exists(PATH_DATA):
            print('coucou2')
            os.mkdir(PATH_DATA)
            os.mkdir(PATH_DATA_IMAGE)
            os.mkdir(PATH_DATA_VECTEUR)
            os.mkdir(PATH_DATA_CNN)
            os.mkdir(PATH_DATA_CNN_TRAIN)
            os.mkdir(PATH_DATA_CNN_TEST)
            os.mkdir(PATH_DATA_KNN)

            shutil.move(path_image, PATH_DATA_IMAGE)
    else: 
        raise IOError("Ce dossier n'existe pas ! ")


def set_up_model(type_model, name_model):
    """load a model save

    Args:
        type_model (MODEL enum): type model KNN or CNN
        name_model (string): name file model

    Returns:
        Object: model
    """
    model = None
    return model


def train_cnn(train=False, nb_image=0):
    """train CNN with nb_image if != 0 and see performance

    Args:
        train (bool, optional): train model if it's true. Defaults to False.
        nb_image (int, optional): choice nb image for train if !=0. Defaults to 0.
    """
    pass


def train_Knn(train=False):
    """train KNN with nb_image if != 0 and see performance

    Args:
        train (bool, optional): train model if it's true. Defaults to False.
    """


# TODO comprend pas l'interet , a discuter
def print_info(data_picture):
    pass
