import numpy as np


def find_similar_vector(vector):
    """return the vector of the similar image

    Args:
        vector (numpy Array): vector with one line, result of the CNN model

    Returns:
        numpy Array: vector of the similar image in the vector database
    """
    vector = np.array([1, 2, 3, 4, 5, 6])
    return vector


def find_similar_vector_id(vecteur, nb_id):
    """returns a list of id's corresponding to the nb_id of similar vectors

    Args:
        vecteur (numpy Array): vector with one line, result of the CNN model
        nb_id (int): desired id number, error if nb_id = 0

    Returns:
        list(int): list of id image
    """
    id = [1, 2, 4, 5, 6, 8]
    return id


def test_performance_model(model):
    """displays the performance of the model with different metrics, using graphs

    Args:
        model (model Object): already trained model that you want to test
    """
    pass


def train_model(nb_image, hp_model):
    """train model cnn with its hyperparameters

    Args:
        nb_image (int): number image for train
        hp_model (dico): dictionnaire with hyperparameters

    Returns:
        Object model: model train
    """
    model = "CNN"
    return model


def create_model(summary_model):
    """create model which no save

    Args:
        summary_model (dico): model and parameter of model

    Returns:
        Object: model for creation of vector
    """
    model = "CNN"
    return model


def save_model(model):
    """save in a file the name of the model and the parameters after training

    Args:
        model (Object): model object
    """
    pass


def charge_model(PATH_file_model):
    """loads the model with these parameters into a file

    Args:
        PATH_file_model (string): path where model is save

    Returns:
        Object: model object
    """
    return "model"
