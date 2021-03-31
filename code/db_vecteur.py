import numpy as np
import pandas as pd
from keras import backend as K
import os


def new_vectorisation_image_bdd(model_cnn, liste_images):
    """
    Avec K.function on fait passer nos images dans le modèle
    On enregistre les informations de la couche flatten pour chaque image 
    dans liste_images
    """
    liste_vecteurs = []
    func = K.function([model_cnn.input], [model_cnn.layers[-4].output])

    for img in liste_images:
        liste_vecteurs.append(func(img))
  
    return liste_vecteurs


# Renvoie un tuple contenant l'id des images et leurs vecteurs associés 
# bin_path est le chemin du dossier où les images sont stockées
# vecteurs est une liste de np array des images vectorisées
# 

def pointeur_database(vecteurs, PATH_DATA_CNN_TRAIN):
    """Renvoie un tuple contenant l'id des images et leurs vecteurs associés
    On part du principe qu'on exécute les images dans l'ordre , à garder en tête

    Args:
        vecteurs (liste): liste de np array des images vectorisées
        PATH_DATA_CNN_TRAIN (str): chemin du dossier où les images sont stockées

    Returns:
       liste_id_vecteurs (tuple) : id des images et leurs vecteurs associés
    """    

    pictures = [file for file in os.listdir(PATH_DATA_CNN_TRAIN) if file.endswith(('jpg', 'png' ))]

    liste_id_vecteurs = list(zip(pictures, vecteurs))
  
    return liste_id_vecteurs




def find_code(vecteur):
    return 2


def restart_vector(model_cnn, database_image):
    vect1 = np.zeros(10)
    vect2 = np.ones(10)
    df = pd.DataFrame({'col1': vect1, 'col2': vect2})
    return df


def save_base_vector(pointeur_bd_vecteur):
    pass
