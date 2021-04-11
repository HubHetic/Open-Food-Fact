# ===========================================
# IMPORT
# ===========================================

import os
import shutil

# ===========================================
# CHEMIN DES DOSSIERS
# ===========================================
PATH_DATA = "../data"
PATH_DATA_IMAGE = PATH_DATA + "/image"
PATH_DATA_VECTEUR = PATH_DATA + "/vecteur"

PATH_DATA_CNN = PATH_DATA + "/CNN"
PATH_DATA_CNN_TRAIN = PATH_DATA_CNN + "/TRAIN"
PATH_DATA_CNN_TEST = PATH_DATA_CNN + "/TEST"

PATH_DATA_IMAGE_SCRAP = PATH_DATA + "/Data_a_scrap"

PATH_DATA_KNN = PATH_DATA + "/KNN"

# ===========================================
# CHEMIN DES FICHIERS
# ===========================================

PATH_DATA_CNN_VG16 = PATH_DATA_CNN + "/vg16.txt"
PATH_DATA_VECTEUR_FILE = PATH_DATA_VECTEUR + "/vector.csv"
PATH_DATA_TRAIN = PATH_DATA_IMAGE_SCRAP + "/data_url_image_train_110k_v2.csv"
PATH_DATA_TEST = PATH_DATA_IMAGE_SCRAP + "/data_url_image_test_110k_v2.csv"


# ===========================================
# FONCTION
# ===========================================

def implement(path_image):
    """setting up all the folders, files, database for the production,
        training of CNN, KNN

        Args:
        path_image (str): chemin absolu vers le dossier des images que
        l'on veut rentrer
        dans le modèle
    """
    if not os.path.exists(PATH_DATA):
        os.mkdir(PATH_DATA)
    if not os.path.exists(PATH_DATA_IMAGE):
        os.mkdir(PATH_DATA_IMAGE)
    if not os.path.exists(PATH_DATA_VECTEUR):
        os.mkdir(PATH_DATA_VECTEUR)
    if not os.path.exists(PATH_DATA_CNN):
        os.mkdir(PATH_DATA_CNN)
    if not os.path.exists(PATH_DATA_KNN):
        os.mkdir(PATH_DATA_KNN)
    if path_image == '':
        return None
    elif os.path.exists(path_image):
        lis_img = os.listdir(path_image)
        for im in lis_img:
            shutil.move(path_image + "/" + im, PATH_DATA_IMAGE)
    else:
        raise IOError("Ce dossier n'existe pas ! ")
