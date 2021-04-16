# ===========================================
# IMPORT
# ===========================================
import os
import time
import matplotlib.pyplot as plt
import pandas as pd
import random

import file_variable as fv
from cnn_file import CNN
from knn_file import class_knn
from db_vecteur import new_list_vecteur_bdd
from db_vecteur import save_base_vector


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


def picture_to_pictures(path_picture, nb_pictures):
    """return the list of paths of nb_pictures similar to the image stored
    in the image path

    Args:
        - path_picture (str): path of the image file to test
        - nb_pictures (int): number of similar images we want to return

    Returns:
        list(str): list of image file paths
    """
    list_id = picture_to_list_code(path_picture, nb_pictures)
    return [MODEL_CNN.find_image(code) for code in list_id]


def picture_to_list_code(path_picture, nb_pictures):
    """return the ids nb_pictures similar to the image
    stored in the path_picture

    Args:
        - path_picture (str): path of the image file to test
        - nb_pictures (int): number of image ids we want to return

    Returns:
        list(str): list of id similar images
    """
    vec = MODEL_CNN.image_to_vector(path_picture)
    return MODEL_KNN.find_similar_vector_id(vec, nb_pictures)


def find_path_sim_picture_to_code(code):
    name = DF_PRODUIT_TEST.loc[code, 'product_name']
    index = (DF_PRODUIT_TRAIN[DF_PRODUIT_TRAIN['product_name'] == name].index)[0]
    return MODEL_CNN.find_image(index)


def show_image(path_picture, nb_pictures):
    """displays the image in the path and the nb_pictures similar to this image

    Args:
        - path_picture (str): path of the image file to display
        - nb_pictures (int): number of similar images to be displayed
    """
    print("=============================")
    vector = MODEL_CNN.changer_format(path_picture, (224, 224))[0]
    code = path_picture.split("/")[-1].replace(".jpg", "")
    path_picture_train = find_path_sim_picture_to_code(code)
    vector2 = MODEL_CNN.changer_format(path_picture_train, (224, 224))[0]
    fig, axes = plt.subplots(1, 2, figsize=(4, 2))
    axes[0].imshow(vector)
    axes[0].legend("picture test")
    axes[1].imshow(vector2)
    axes[1].legend("picture of similar product picture test")
    list_image_found = picture_to_pictures(path_picture, nb_pictures)
    fig2, axes = plt.subplots(1, nb_pictures, figsize=(2 * nb_pictures, 2))
    for img, ax in zip(list_image_found, axes):
        vector = MODEL_CNN.changer_format(img, (224, 224))[0]
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
    MODEL_CNN.charge_model()
    MODEL_KNN = class_knn()
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
    MODEL_KNN.charge_database(path)


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
        MODEL_KNN.charge_model(name_model=name_model)
    else:
        MODEL_CNN.charge_model(name_model=name_model)


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


def train_cnn(nb_pictures=0, format=(224, 224), verbose=False):
    """Trains the model stored in the variable CNN_MODEL with nb_pictures,
    with the images preprocessed with the variable format to define the
    format of the image

    Args:
        - nb_pictures (int, optional): number of images for training the model
        if nb_pictures = 0 all images stored in data/image are used.
        Defaults to 0.
        - format ((int, int)), optional): image format after preprocessing.
        Defaults to (224, 224).
        - verbose (bool, optional): displays the different steps performed
        and the calculation time. Defaults to False.

    Exemple:
        # train the model on all images
        >>> train_cnn()
        # train the model on 2000 images, format (128, 128)
        >>> train_cnn(nb_pictures=2000, format=(128, 128))
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
        list_pictures, format)
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


def create_dataframe_vector(list_name_picture):
    img_prep, name = MODEL_CNN.changer_format_image_folder(list_name_picture,
                                                           (224, 224))
    return new_list_vecteur_bdd(MODEL_CNN.MODEL, img_prep, [name])


def create_database_vectorize(name_database="vector.csv", verbose=False):
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
    df = create_dataframe_vector([list_pictures.pop(0)])
    index = 0
    for new_index in range(5000, len(list_pictures), 5000):
        reduce_list_image = list_pictures[index:new_index]
        df = pd.concat([df, create_dataframe_vector(reduce_list_image)])
        index = new_index
        if verbose:
            tps2 = time.time()
            print(f"number processed: {new_index}")
            print(f"Execution time: {tps2 - tps1}")
    reduce_list_image = list_pictures[index:]
    df = pd.concat([df, create_dataframe_vector(reduce_list_image)])
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


def code_to_name_produit(liste_id):
    return [DF_PRODUIT_TRAIN.loc[id, 'product_name'] for id in liste_id]


def test_performance_cnn(nb_picturess_test=5, verbose=False):
    """[summary]

    Args:
        - nb_picturess_test (int, optional): [description]. Defaults to 5.
        - verbose (bool, optional): [afficher les différents étapes
        réalisé et le temps de calcul]. Defaults to False.

    """
    if verbose:
        nb_pictures_done = 0
        print("=========================")
        print("Start create new dataset")
        tps1 = time.time()
    images_a_tester = os.listdir(fv.PATH_DATA_CNN_TEST)
    list_pictures = [fv.PATH_DATA_CNN_TEST+image for image in images_a_tester]
    images_a_tester = [image.replace('.jpg', '') for image in images_a_tester]
    table_index = dict([(i, 0) for i in range(nb_picturess_test)])
    table_index[-1] = 0
    for path_picture, code_origin_image in zip(list_pictures, images_a_tester):
        if verbose:
            if (nb_pictures_done % 300) == 0:
                tps2 = time.time()
                print(f"number processed: {nb_pictures_done}")
                print(f"Execution time: {tps2 - tps1}")
            nb_pictures_done += 1
        try:
            liste_id = picture_to_list_code(path_picture, nb_picturess_test)
        except:
            continue
        try:
            liste_produits = code_to_name_produit(liste_id)
        except KeyError as e:
            print(f"code produit not found : {e}")
            continue
        nom_du_produit = DF_PRODUIT_TEST.loc[code_origin_image, 'product_name']
        try:
            table_index[liste_produits.index(nom_du_produit)] += 1
        except ValueError:
            table_index[-1] += 1
    x = [i for i in range(1, nb_picturess_test + 2)]
    y = [table_index[i] for i in range(nb_picturess_test)]
    y.append(table_index[-1])
    plt.bar(x, y)
