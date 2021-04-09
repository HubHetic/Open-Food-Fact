import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from file_variable import PATH_DATA_IMAGE


def changer_format(path_image, format):
    """changer le format de l'image dans path_image et la préprocess pour le CNN

    Args:
        path_image (str): chemin d'accès de l'image
        format (tuple(int,int)): format d'image

    Returns:
        [np array]: numpy array de l'image préprocess
    
    Exemple:
        # preproces image au format (224, 224)
        >>> path_image = "./data/nutella.jpg"
        >>> img_prep = changer_format(path_image, (224, 224))
        >>> img_prep.shape
            (1, 224, 224, 3)
    """
    img = image.load_img(path_image, color_mode='rgb', target_size=format)
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img_prep = preprocess_input(img)
    return img_prep


def find_image(code_image):
    """retourne le chemin de l'image dont le nom est code_image

    Args:
        code_image (string): id image

    Returns:
        string: chemin relatif de l'image stocké dans le dossier data/image
    """
    path_image = PATH_DATA_IMAGE + '/' + code_image + '.jpg'
    return path_image


def changer_format_image_folder(list_image, format):
    """préprocess les images stocké dans list_image avec format

    Args:
        list_image (list(string)): nom des différents image
        format ((int, int)): format final de l'image preprocess

    Returns:
        (numpy.array, list(string)): le couple images préprocess et le 
        nom des images
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
