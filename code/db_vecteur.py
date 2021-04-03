import numpy as np
import pandas as pd
from keras import backend as bk
import os
from file_variable import PATH_DATA_VECTEUR_FILE


def image_to_vector(func, image):
    return func(image)[0][0]


def function_last_layers(model_cnn):
    return bk.function([model_cnn.input], [model_cnn.layers[-2].output])


def new_list_vecteur_bdd(model_cnn, liste_images, list_names):
    """
    Avec K.function on fait passer nos images dans le modèle
    On enregistre les informations de la couche flatten pour chaque image
    dans liste_images
    """
    func = function_last_layers(model_cnn)
    vector_list = np.array([image_to_vector(func, img) for img in liste_images])
    liste_code = list(map(lambda x: x[:-4], list_names))
    df = create_dataframe_vector(vector_list, liste_code)
    save_base_vector(df, PATH_DATA_VECTEUR_FILE)
    return df


def new_vecteur_to_database(liste_vecteur, liste_name):
    couple = list(zip(liste_name, liste_vecteur))
    np.array(couple)
    return np.array(couple)


# Renvoie un tuple contenant l'id des images et leurs vecteurs associes
# bin_path est le chemin du dossier où les images sont stockées
# vecteurs est une liste de np array des images vectorisées

def pointeur_database(vecteurs, PATH_DATA_CNN_TRAIN):
    """Renvoie un tuple contenant l'id des images et leurs vecteurs associés
    On part du principe qu'on exécute les images dans l'ordre , à garder
    en tête

    Args:
        vecteurs (liste): liste de np array des images vectorisées
        PATH_DATA_CNN_TRAIN (str): chemin du dossier où les images sont
        stockées

    Returns:
       liste_id_vecteurs (tuple) : id des images et leurs vecteurs associés
    """
    pictures = [file for file in os.listdir(PATH_DATA_CNN_TRAIN) if
                file.endswith(('jpg', 'png'))]
    liste_id_vecteurs = list(zip(pictures, vecteurs))
    return liste_id_vecteurs


def find_code(vecteur):
    return 2


def create_dataframe_vector(vector_list, liste_code):
    nb_columns = vector_list.shape[1]
    list_columns = ['vector' + str(i) for i in range(nb_columns)]
    df = pd.DataFrame(vector_list, columns=list_columns)
    df['code'] = pd.Series(liste_code)
    return df


def save_base_vector(pointeur_bd_vecteur, path_save):
    pointeur_bd_vecteur.to_csv(path_save)
