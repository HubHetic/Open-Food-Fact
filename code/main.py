# ===========================================
# IMPORT
# ===========================================
import os
import random
import time
from cnn_file import CNN

# from Knn_file import class_knn
from db_vecteur import find_code, new_list_vecteur_bdd
from db_image import changer_format_folder, find_image
from db_produit import find_line
from file_variable import implement
from file_variable import PATH_DATA_IMAGE


# ===========================================
# VARIABLE GLOBAL
# ===========================================
MODEL_CNN = ''
MODEL_KNN = ''
# ===========================================
# FONCTION
# ===========================================


def image_to_code(image, nb_image):
    """returns the id of a similar image in the variable image

    Args:
        image ([image object]): picture jpg, gz

    Returns:
        int: id code of the image in the databases
    """
    vec = MODEL_CNN.image_to_vector(image)
    vec_sim = MODEL_KNN.find_similar_vector_id(vec, nb_image)
    # code = find_code(vec_sim)
    # return find_image(code)
    return vec_sim


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


def all_implement(path_image, build_model=False):
    implement(path_image)
    if build_model:
        global MODEL_CNN, MODEL_KNN
        MODEL_CNN = CNN()
        MODEL_CNN.charge_model()
        # MODEL_KNN = class_knn()


def set_up_model(type_model, name_model):
    """load a model save

    Args:
        type_model (MODEL enum): type model KNN or CNN
        name_model (string): name file model

    Returns:
        Object: model
    """
    if os.path.exists(name_model):
        type_model.charge_model(PATH_file_model=name_model)
    else:
        type_model.charge_model(name_model=name_model)


def train_cnn(nb_image=0, format=(224, 224), verbose=False):
    """train CNN with nb_image if != 0 and see performance

    Args:
        train (bool, optional): train model if it's true. Defaults to False.
        nb_image (int, optional): choice nb image for train if !=0. Defaults to 0.
    """
    if verbose:
        print("=========================")
        print("Start")
        tps1 = time.time()
    if nb_image == 0:
        liste_images = os.listdir(PATH_DATA_IMAGE)
    else:
        liste_images = os.listdir(PATH_DATA_IMAGE)
        random.shuffle(liste_images)
        liste_images = liste_images[0:nb_image]
    if verbose:
        tps2 = time.time()
        print(f"FIND exemple: {len(liste_images)}")
        print(f"temps d'execution: {tps2 - tps1}")
        print("=========================")
        print("Start image prep")
        tps1 = time.time()
    list_images_prep, list_names = changer_format_folder(liste_images, format)
    if verbose:
        tps2 = time.time()
        print(f"temps d'execution: {tps2 - tps1}")
        print(f"len of list_image_prep : {len(list_images_prep)}")
        print(f"shape one image prep : {list_images_prep[0].shape}")
        print("=========================")
        print("Start Train model")
        tps1 = time.time()
    MODEL_CNN.train_model(list_images_prep)
    if verbose:
        tps2 = time.time()
        print(f"temps d'execution: {tps2 - tps1}")
        print("=========================")
        print("Start create new dataset")
        tps1 = time.time()
    liste_vecteur = new_list_vecteur_bdd(MODEL_CNN.MODEL, list_images_prep, list_names)
    if verbose:
        tps2 = time.time()
        print(f"temps d'execution: {tps2 - tps1}")
    return liste_vecteur


def train_Knn(train=False):
    """train KNN with nb_image if != 0 and see performance

    Args:
        train (bool, optional): train model if it's true. Defaults to False.
    """


# TODO comprend pas l'interet , a discuter
def print_info(data_picture):
    pass
