# ===========================================
# IMPORT
# ===========================================
from cnn_file import charge_model, image_to_vector
from knn_file import find_similar_vector_id, charge_model_cluster
from knn_file import train_model
from db_vecteur import find_code, new_list_vecteur_bdd
from db_image import changer_format_folder, find_image
from db_produit import find_line
from file_variable import implement
from file_variable import PATH_DATA_IMAGE

# ===========================================
# FONCTION
# ===========================================


def image_to_code(image):
    """returns the id of a similar image in the variable image

    Args:
        image ([image object]): picture jpg, gz

    Returns:
        int: id code of the image in the databases
    """
    vec = image_to_vector(image)
    vec_sim = find_similar_vector_id(vec)
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


def all_implement(path_image, build_vecteur=False):
    implement(path_image)


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
    model = charge_model('')
    list_images, list_names = changer_format_folder(PATH_DATA_IMAGE)
    liste_vecteur = new_list_vecteur_bdd(model, list_images, list_names)
    # l = new_vecteur_to_database(liste_vecteur, list_images)
    return liste_vecteur


def train_Knn(train=False):
    """train KNN with nb_image if != 0 and see performance

    Args:
        train (bool, optional): train model if it's true. Defaults to False.
    """
    model = charge_model_cluster()
    train_model()


# TODO comprend pas l'interet , a discuter
def print_info(data_picture):
    pass
