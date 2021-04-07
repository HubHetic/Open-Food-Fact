from PIL import Image
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
from file_variable import PATH_DATA_IMAGE


def changer_format(path_image, format):
    """changer le format de l'image passé en paramètre et la préprocess pour le CNN

    Args:
        path_image (str): chemin d'accès de l'image
        format (tuple(int,int)): format d'image

    Returns:
        [np array]: numpy array de l'image préprocess
    """
    img = image.load_img(path_image, color_mode='rgb', target_size=format)
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img_prep = preprocess_input(img)
    return img_prep


def find_image(code_image):
    image = Image.open(PATH_DATA_IMAGE + '/' + code_image + '.jpg')
    return image


def changer_format_folder(list_image, format):
    """préprocess toutes les images d'un dossier

    Args:
        bin_path (str): chemin du dossier

    Returns:
        [liste]: liste de np array des images du dossier préprocess
    """

    liste_img_prep = []
    FORMAT = (224, 224)
    for img in list_image:
        path = PATH_DATA_IMAGE + "/" + img
        try:
            liste_img_prep.append(changer_format(path, FORMAT))
        except:
            continue
    return liste_img_prep, list_image
