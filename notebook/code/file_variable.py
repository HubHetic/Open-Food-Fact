# ===========================================
# IMPORT
# ===========================================

import os
import shutil

# ===========================================
# CHEMIN DES DOSSIERS
# ===========================================
path = os.path.abspath(".")

PATH_ORIGINE = path[:path.find("Open-Food-Fact") + len("Open-Food-Fact")]
PATH_ORIGINE += '/'

PATH_DATA = PATH_ORIGINE + "data/"
PATH_DATA_IMAGE = PATH_DATA + "image/"
PATH_DATA_VECTOR = PATH_DATA + "vector/"

PATH_DATA_CNN_TRAIN = PATH_DATA_IMAGE + "TRAIN/"
PATH_DATA_CNN_TEST = PATH_DATA_IMAGE + "TEST/"

PATH_DATA_PICTURE_SCRAP = PATH_DATA + "Data_a_scrap/"

PATH_DATA_KNN = PATH_DATA + "KNN/"

# ===========================================
# CHEMIN DES FICHIERS
# ===========================================

PATH_DATA_VECTOR_FILE = PATH_DATA_VECTOR + "vector.csv"
PATH_DATA_TRAIN = PATH_DATA_PICTURE_SCRAP + "data_image_train.csv"
PATH_DATA_TEST = PATH_DATA_PICTURE_SCRAP + "data_image_test.csv"


# ===========================================
# FONCTION
# ===========================================

def implement(path_image):
    """setting up all the folders, files, database for the production,
        training of CNN, KNN

        Args:
        path_image (str): absolute path to the folder of the
        images you want to enter in the template
    """
    if not os.path.exists(PATH_DATA):
        os.mkdir(PATH_DATA)
    if not os.path.exists(PATH_DATA_IMAGE):
        os.mkdir(PATH_DATA_IMAGE)
    if not os.path.exists(PATH_DATA_VECTOR):
        os.mkdir(PATH_DATA_VECTOR)
    if not os.path.exists(PATH_DATA_KNN):
        os.mkdir(PATH_DATA_KNN)
    if not os.path.exists(PATH_DATA_CNN_TEST):
        os.mkdir(PATH_DATA_CNN_TEST)
    if path_image == '':
        return None
    elif os.path.exists(path_image):
        lis_img = os.listdir(path_image)
        for im in lis_img:
            shutil.move(path_image + im, PATH_DATA_IMAGE)
    else:
        raise IOError("This file does not exist !")
