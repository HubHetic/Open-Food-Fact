# ===========================================
# IMPORT
# ===========================================

import numpy as np
import pandas as pd
from keras import backend as bk
# ===========================================
# FONCTION
# ===========================================


def image_vector_CNN(func, image):
    return func(image)[0][0]


def function_last_layers(model_cnn):
    return bk.function([model_cnn.input], [model_cnn.layers[-2].output])


def new_list_vector_bdd(model_cnn, list_images, list_names):
    """
    Avec K.function on fait passer nos images dans le modèle
    On enregistre les informations de la couche flatten pour chaque image
    dans list_images
    """
    func = function_last_layers(model_cnn)
    vector_list = np.array([image_vector_CNN(func, x) for x in list_images])
    liste_code = list(map(lambda x: x[:-4], list_names))
    return create_dataframe_vector(vector_list, liste_code)


def new_vector_to_database(list_vector, list_names):
    couple = list(zip(list_names, list_vector))
    np.array(couple)
    return np.array(couple)


def create_dataframe_vector(vector_list, liste_code):
    nb_columns = vector_list.shape[1]
    list_columns = ['vector' + str(i) for i in range(nb_columns)]
    df = pd.DataFrame(vector_list, columns=list_columns)
    df['code'] = pd.Series(liste_code)
    return df


def save_base_vector(pointeur_bd, path_save):
    pointeur_bd.to_csv(path_save, index=False)
