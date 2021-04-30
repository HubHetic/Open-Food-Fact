# ===========================================
# IMPORT
# ===========================================
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random

import file_variable as fv
from cnn_file import CNN
from knn_file import ClassKnn
from db_vector import new_list_vector_bdd
from db_vector import save_base_vector


# ===========================================
# VARIABLE GLOBALE
# ===========================================
MODEL_CNN = ''
MODEL_KNN = ''

DF_PRODUIT_TRAIN = pd.read_csv(fv.PATH_DATA_TRAIN)[['code', 'product_name']]
DF_PRODUIT_TRAIN = DF_PRODUIT_TRAIN.set_index('code')

DF_PRODUIT_TEST = pd.read_csv(fv.PATH_DATA_TEST)[['code', 'product_name']]
DF_PRODUIT_TEST = DF_PRODUIT_TEST.set_index('code')
# ===========================================
# FONCTION
# ===========================================


def picture_to_pictures(path_picture, nb_pictures, size):
    """return the list of paths of nb_pictures similar to the image stored
    in the image path

    Args:
        - path_picture (str): path of the image file to test
        - nb_pictures (int): number of similar images we want to return

    Returns:
        list(str): list of image file paths
    """
    list_id = picture_to_list_code(path_picture, nb_pictures, size)
    return [MODEL_CNN.find_image(code) for code in list_id]


def picture_to_list_code(path_picture, nb_pictures, size):
    """return the ids nb_pictures similar to the image
    stored in the path_picture

    Args:
        - path_picture (str): path of the image file to test
        - nb_pictures (int): number of image ids we want to return

    Returns:
        list(str): list of id similar images
    """
    vec = MODEL_CNN.image_to_vector(path_picture, size)
    return MODEL_KNN.find_similar_vector_id(vec, nb_pictures)


def find_path_sim_picture_to_code(code):
    name = DF_PRODUIT_TEST.loc[code, 'product_name']
    index = (DF_PRODUIT_TRAIN[DF_PRODUIT_TRAIN['product_name'] == name].index)[0]
    return MODEL_CNN.find_image(index)


def show_image(path_picture, nb_pictures, size=(224, 224)):
    """displays the image in the path and the nb_pictures similar to this image

    Args:
        - path_picture (str): path of the image file to display
        - nb_pictures (int): number of similar images to be displayed
    """
    print("=============================")
    vector = MODEL_CNN.changer_format(path_picture, size)[0]
    code = path_picture.split("/")[-1].replace(".jpg", "")
    path_picture_train = find_path_sim_picture_to_code(code)
    vector2 = MODEL_CNN.changer_format(path_picture_train, size)[0]
    fig, axes = plt.subplots(1, 2, figsize=(4, 2))
    axes[0].imshow(vector)
    axes[0].legend("picture test")
    axes[1].imshow(vector2)
    axes[1].legend("picture of similar product picture test")
    list_image_found = picture_to_pictures(path_picture, nb_pictures, size)
    fig2, axes = plt.subplots(1, nb_pictures, figsize=(2 * nb_pictures, 2))
    for img, ax in zip(list_image_found, axes):
        vector = MODEL_CNN.changer_format(img, size)[0]
        ax.imshow(vector)
        ax.axis('off')
    fig2.legend("find pitcture with algo")


def display_data_vector_available():
    """returns the names of the datasets created by the different models

    Returns:
        [string]: list of different names
    """
    return os.listdir(fv.PATH_DATA_VECTOR)


def all_implement(path_picture):
    """implement the different folders necessary to deploy the project,
    move the images from the path_picture to the data/image folder to
    be able to test them. instantiate the MODEL_CNN and MODEL_KNN classes

    - the MODEL_CNN instance allows to choose the model to transform the
    image into a vector as well as the different training and performance
    test functions.

    - the instance MODEL_KNN allows to choose the model to select the similar
    images.

    if you have already put the images in the folder data/image/ put only
    '' for the path_picture

    Args:
        path_picture (string): absolute path of the folder containing the
        images to be moved to the data/image folder
    """
    fv.implement(path_picture)
    global MODEL_CNN, MODEL_KNN
    MODEL_CNN = CNN()
    MODEL_CNN.load_model()
    MODEL_KNN = ClassKnn()
    MODEL_KNN.charge_model()
    print(MODEL_KNN.model)


def choice_vector_database(name_database):
    """Choose the vector dataset used by the model_KNN
    to search for similar images

    Args:
        - name_database (string): name of the dataset to be used,
        the list of names can be found with the function
        display_data_vector_available
    """
    path = fv.PATH_DATA_VECTOR + name_database
    MODEL_KNN.load_knn_df(path)


def set_up_model(type_model, name_model):
    """Load the model name_model into the type_model which is either KNN or CNN

    Args:
        - type_model (string): value "KNN" ou "CNN"
        - name_model (string): name file model

    exemple:
        # change model VGG16 par Mobilenet
        >>> set_up_model("CNN", "mobile_net")
        # remove on the modele VGG16
        >>> set_up_model("CNN", "VGG16")
    """
    if type_model == "KNN":
        MODEL_KNN.load_model(name_model=name_model)
    else:
        MODEL_CNN.load_model(name_model=name_model)


def train_knn(name_database, verbose=False):
    choice_vector_database(name_database)
    if verbose:
        print("=========================")
        print("Start")
        print(f"dataset size: {MODEL_KNN.df_vecteur.shape}")
        tps1 = time.time()
    MODEL_KNN.train()
    if verbose:
        tps2 = time.time()
        print(f"Execution time: {tps2 - tps1}")
        print("=========================")


def train_cnn(nb_pictures=0, size=(224, 224), verbose=False):
    """Trains the model stored in the variable CNN_MODEL with nb_pictures,
    with the images preprocessed with the variable size to define the
    size of the image

    Args:
        - nb_pictures (int, optional): number of images for training the model
        if nb_pictures = 0 all images stored in data/image are used.
        Defaults to 0.
        - size ((int, int)), optional): image size after preprocessing.
        Defaults to (224, 224).
        - verbose (bool, optional): displays the different steps performed
        and the calculation time. Defaults to False.

    Exemple:
        # train the model on all images
        >>> train_cnn()
        # train the model on 2000 images, size (128, 128)
        >>> train_cnn(nb_pictures=2000, size=(128, 128))
        # train the model on all images and display
        # calculation time
        >>> train_cnn(verbose=True)
    """
    if verbose:
        print("=========================")
        print("Start")
        tps1 = time.time()
    list_pictures = os.listdir(fv.PATH_DATA_IMAGE)
    if nb_pictures != 0:
        random.shuffle(list_pictures)
        list_pictures = list_pictures[0:nb_pictures]
    if verbose:
        tps2 = time.time()
        print(f"Find exemple: {len(list_pictures)}")
        print(f"Execution time: {tps2 - tps1}")
        print("=========================")
        print("Start image prep")
        tps1 = time.time()
    list_pictures_prep, list_names = MODEL_CNN.changer_format_image_folder(
        list_pictures, size)
    if verbose:
        tps2 = time.time()
        print(f"Execution time: {tps2 - tps1}")
        print(f"Len of list_image_prep : {len(list_pictures_prep)}")
        print(f"Shape one image prep : {list_pictures_prep[0].shape}")
        print("=========================")
        print("Start Train model")
        tps1 = time.time()
    MODEL_CNN.train_model(list_pictures_prep)
    if verbose:
        tps2 = time.time()
        print(f"Execution time: {tps2 - tps1}")
        print("=========================")


def create_dataframe_vector(list_name_picture, size):
    img_prep, name = MODEL_CNN.changer_format_image_folder(list_name_picture,
                                                           size)
    return new_list_vector_bdd(MODEL_CNN.MODEL, img_prep, [name])


def create_database_vectorize(name_database="vector.csv", verbose=False,
                              size=(224, 224)):
    """constructs a vector dataset of all the images stored in the data/image
    folder with the CNN model used in the MODEL_CNN variable and stores it in
    the data/vector folder

    Args:
        - name_database=(bool, optional): [database name].
         Defaults to "vector.csv"
        - verbose (bool, optional): [Display the different steps performed
        and the calculation time]. Defaults to False.
    """
    list_pictures = os.listdir(fv.PATH_DATA_IMAGE)
    if verbose:
        print("=========================")
        print("Start create new dataset")
        tps1 = time.time()
    df = create_dataframe_vector([list_pictures.pop(0)], size)
    index = 0
    for new_index in range(5000, len(list_pictures), 5000):
        reduce_list_image = list_pictures[index:new_index]
        df = pd.concat([df, create_dataframe_vector(reduce_list_image, size)])
        index = new_index
        if verbose:
            tps2 = time.time()
            print(f"number processed: {new_index}")
            print(f"Execution time: {tps2 - tps1}")
    reduce_list_image = list_pictures[index:]
    df = pd.concat([df, create_dataframe_vector(reduce_list_image, size)])
    index = new_index
    if verbose:
        tps2 = time.time()
        print(f"number processed: {len(list_pictures)}")
        print(f"Execution time: {tps2 - tps1}")
    df = df.iloc[1:]
    if name_database == "vector.csv":
        save_base_vector(df, fv.PATH_DATA_VECTOR_FILE)
    else:
        save_base_vector(df, fv.PATH_DATA_VECTOR + name_database)


def code_to_name_product(liste_id, label):
    return [DF_PRODUIT_TRAIN.loc[id, label] for id in liste_id]


def performance_test_cnn(nb_pictures_test=15, label='product_name',
                         verbose=False, size=(224, 224)):
    """for all then pictures in test, we ll apply knn
        and compt all the occurence where our pictures tested has
        the same name

    Args:
        - nb_picturess_test (int, optional): [description]. Defaults to 5.
        - verbose (bool, optional): [display the different steps
        performed and the calculation time]. Defaults to False.

    """
    if verbose:
        nb_pictures_done = 0
        print("=========================")
        print("Start create new dataset")
        tps1 = time.time()
    test_image = os.listdir(fv.PATH_DATA_CNN_TEST)[:600]
    list_pictures = [fv.PATH_DATA_CNN_TEST+image for image in test_image]
    test_image = [image.replace('.jpg', '') for image in test_image]
    table_index = dict([(i, 0) for i in range(nb_pictures_test)])
    table_index[-1] = 0
    for path_picture, code_origin_image in zip(list_pictures, test_image):
        if verbose:
            if (nb_pictures_done % 300) == 0:
                tps2 = time.time()
                print(f"number processed: {nb_pictures_done}")
                print(f"Execution time: {tps2 - tps1}")
            nb_pictures_done += 1

        try:
            liste_id = picture_to_list_code(path_picture, nb_pictures_test,
                                            size)
        except:
            continue
        try:
            liste_produits = code_to_name_product(liste_id, label)

        except KeyError as e:
            print(f"code product not found : {e}")
            continue
        name_product = DF_PRODUIT_TEST.loc[code_origin_image, label]
        try:
            table_index[liste_produits.index(name_product)] += 1
        except ValueError:
            table_index[-1] += 1
    x = [i for i in range(1, nb_pictures_test + 1)]
    len_test = len(test_image)
    for i in range(1, nb_pictures_test):
        table_index[i] += table_index[i - 1]
    y = [(table_index[i] / len_test) * 100 for i in range(nb_pictures_test)]
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.bar(x, y)
    grid_x_ticks = np.arange(0, 25, 0.2)
    grid_y_ticks = np.arange(0, 5, 0.2)
    ax.set_xticks(grid_x_ticks, minor=True)
    ax.set_yticks(grid_y_ticks, minor=True)

    ax.grid(which='both')
    ax.grid(which='minor', alpha=0.2, linestyle='--')
    ax.set_ylabel('percentage')
    ax.set_title('percentage image find by ' + label + ' with '
                 + str(nb_pictures_test))
