# ===========================================
# IMPORT
# ===========================================
import os
import time
from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt
from cnn_file import CNN
from knn_file import class_knn
from db_vecteur import new_list_vecteur_bdd, save_base_vector
from db_produit import find_line
from file_variable import implement, PATH_DATA_VECTEUR
from file_variable import PATH_DATA_IMAGE, PATH_DATA_VECTEUR_FILE
from file_variable import PATH_DATA_TRAIN, PATH_DATA_TEST, PATH_DATA_CNN_TEST
import pandas as pd
import random


# ===========================================
# VARIABLE GLOBALE
# ===========================================
MODEL_CNN = ''
MODEL_KNN = ''

DF_PRODUIT_TRAIN = pd.read_csv(PATH_DATA_TRAIN)[['code', 'product_name']]
DF_PRODUIT_TRAIN = DF_PRODUIT_TRAIN.set_index('code')

DF_PRODUIT_TEST = pd.read_csv(PATH_DATA_TEST)[['code', 'product_name']]
DF_PRODUIT_TEST = DF_PRODUIT_TEST.set_index('code')
# ===========================================
# FONCTION
# ===========================================


def image_to_images(path_image, nb_image):
    """retourne la liste des chemins des nb_images similaire à l'image
    stocker dans le path image

    Args:
        path_image (str): chemin du fichier image à tester
        nb_image (int): nombre d'image simmilaire qu'on veut retourner

    Returns:
        list(str): liste des chemins des fichiers images
    """
    list_id = image_to_code(path_image, nb_image)
    list_image_path = [MODEL_CNN.find_image(code) for code in list_id]
    return list_image_path


def image_to_code(path_image, nb_image):
    """return les id nb_image similaires à l'image stocker
    dans le chemins path_image

    Args:
        path_image (str): chemin du fichier image à tester
        nb_image (int): nombre d'id image qu'on veut retourner

    Returns:
        list(str): liste des id images similaires
    """
    vec = MODEL_CNN.image_to_vector(path_image)
    return MODEL_KNN.find_similar_vector_id(vec, nb_image)


def show_image(path_image, nb_image):
    """affiche l'image dans le chemin path et les nb_image similaires
    à cette image

    Args:
        path_image (str): chemin du fichier image à afficher
        nb_image (int): nombre d'image similaire que l'on veut afficher
    """
    print("=============================")
    print(f"image a trouver {path_image.split(',')[-1]}")
    vect = MODEL_CNN.changer_format(path_image, (224, 224))[0]
    print(vect.shape)
    fig, axes = plt.subplots(1, 1, figsize=(2, 2))
    imshow(vect)
    list_image_found = image_to_images(path_image, nb_image)
    fig2, axes = plt.subplots(1, nb_image, figsize=(2 * nb_image, 2))
    for img, ax in zip(list_image_found, axes):
        vect = MODEL_CNN.changer_format(img, (224, 224))[0]
        ax.imshow(vect)
        ax.axis('off')


def image_to_line_data(image):
    """returns the product line of the image variable in the product database

    Args:
        image (Image Object): image.jpg gz

    Returns:
        pandas.Series: line in the product database
    """
    # code = image_to_code(image, 1)
    line = find_line()
    return line


def display_data_vector_available():
    """retourne les noms des datasets créer par les différent modèle

    Returns:
        [string]: liste des différents noms
    """
    return os.listdir(PATH_DATA_VECTEUR)


def all_implement(path_image):
    """implement les différent dossier nécessaire au déploiment du projet,
    déplacer les images du path_image vers le dossiers data/image pour
    pouvoir les tester. instancie les class MODEL_CNN et MODEL_KNN

    - l'instance MODEL_CNN permet de choisir le modele pour transformer
    l'image en vecteur ainsi que les différents fonction d'entrainement
    et de teste de performances.

    - l'instance MODEL_KNN permet de choisir le modele pour choisir
    les images similaire

    si vous avez déjà mis les images dans le dossier data/image/
    mettait seulement  '' pour le path_image

    Args:
        path_image (string): chemin absolue du dossier contenant
        les images à déplacer dans le dossier data/image
    """
    implement(path_image)
    global MODEL_CNN, MODEL_KNN
    MODEL_CNN = CNN()
    MODEL_CNN.charge_model()
    MODEL_KNN = class_knn()
    MODEL_KNN.charge_model()
    print(MODEL_KNN.model)


def choice_vector_database(name_database):
    """choisi le dataset vecteur utilisé par le model_KNN
    pour la recherche des images similaires

    Args:
        name_database (string): nom du datasets à utilisé,
        la liste des noms peut se trouver avec la fonction
        display_data_vector_available
    """
    path = PATH_DATA_VECTEUR + '/' + name_database
    MODEL_KNN.charge_database(path)


def set_up_model(type_model, name_model):
    """charger le modele name_model dans le type_model qui est soit
    KNN ou CNN

    Args:
        type_model (string): valeur "KNN" ou "CNN"
        name_model (string): name file model

    exemple:
        # changer model VGG16 par Mobilenet
        >>> set_up_model("CNN", "mobile_net")
        # revenir sur le modele VGG16
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
        print(f"taille du dataset: {MODEL_KNN.df_vecteur.shape}")
        tps1 = time.time()
    MODEL_KNN.train()
    if verbose:
        tps2 = time.time()
        print(f"temps d'execution: {tps2 - tps1}")
        print("=========================")


def train_cnn(nb_image=0, format=(224, 224), verbose=False):
    """entraine le modele stocké dans la variable CNN_MODEL avec nb_image,
    avec les images préprocessées avec la variable format pour définir
    le format de l'image

    Args:
        nb_image (int, optional): nombre d'image pour l'entrainement du model
         si nb_image = 0 toutes les images stockées dans data/image sont
         utilisées
         . Defaults to 0.
        format ((int, int)), optional): format de l'image après le
        préprocessing. Defaults to (224, 224).
        verbose (bool, optional): affiche les différents étapes réaliées
        et le temps de calcul. Defaults to False.

    Exemple:
        # entrainer le modele sur toutes les images
        >>> train_cnn()
        # entrainer le modele sur 2000 images, au format (128, 128)
        >>> train_cnn(nb_image=2000, format=(128, 128))
        # entrainer le modele sur toutes les images et affiché
        # le temps de calcul
        >>> train_cnn(verbose=True)
    """
    if verbose:
        print("=========================")
        print("Start")
        tps1 = time.time()
    if nb_image == 0:
        liste_images = os.listdir(PATH_DATA_IMAGE)
    else:
        liste_images = os.listdir(PATH_DATA_IMAGE)
        random.shuffle(liste_images)
        liste_images = liste_images[0:nb_image]
    if verbose:
        tps2 = time.time()
        print(f"FIND exemple: {len(liste_images)}")
        print(f"temps d'execution: {tps2 - tps1}")
        print("=========================")
        print("Start image prep")
        tps1 = time.time()
    list_images_prep, list_names = MODEL_CNN.changer_format_image_folder(
        liste_images, format)
    if verbose:
        tps2 = time.time()
        print(f"temps d'execution: {tps2 - tps1}")
        print(f"len of list_image_prep : {len(list_images_prep)}")
        print(f"shape one image prep : {list_images_prep[0].shape}")
        print("=========================")
        print("Start Train model")
        tps1 = time.time()
    MODEL_CNN.train_model(list_images_prep)
    if verbose:
        tps2 = time.time()
        print(f"temps d'execution: {tps2 - tps1}")
        print("=========================")


def vectoriser(name_database="vector.csv", verbose=False):
    """contruit un dataset vecteur de l'ensemble des images stocker
    dans le dossier data/image avec le modele CNN utilisé dans
    la variable MODEL_CNN et le stocker dans le dossier data/vecteur

    Args:
        verbose (bool, optional): [afficher les
        différents étapes réalisé et le temps de calcul]. Defaults to False.
    """
    liste_images = os.listdir(PATH_DATA_IMAGE)
    if verbose:
        print("=========================")
        print("Start create new dataset")
        tps1 = time.time()
    img = [liste_images.pop(0)]
    img_prep, name = MODEL_CNN.changer_format_image_folder(img, (224, 224))
    df = new_list_vecteur_bdd(MODEL_CNN.MODEL, img_prep, [name])
    index = 0
    for new_index in range(5000, len(liste_images), 5000):
        reduce_list_image = liste_images[index:new_index]
        list_img_prep, list_names = MODEL_CNN.changer_format_image_folder(
            reduce_list_image, (224, 224))
        df = pd.concat([df, new_list_vecteur_bdd(MODEL_CNN.MODEL,
                                                 list_img_prep,
                                                 list_names)])
        index = new_index
        if verbose:
            tps2 = time.time()
            print(f"nombre traité: {new_index}")
            print(f"temps d'execution: {tps2 - tps1}")
    reduce_list_image = liste_images[index:]
    list_img_prep, list_names = MODEL_CNN.changer_format_image_folder(
        reduce_list_image, (224, 224))
    df = pd.concat([df, new_list_vecteur_bdd(MODEL_CNN.MODEL, list_img_prep,
                                             list_names)])
    index = new_index
    if verbose:
        tps2 = time.time()
        print(f"nombre traité: {len(liste_images)}")
        print(f"temps d'execution: {tps2 - tps1}")
    df = df.iloc[1:]
    if name_database == "vector.csv":
        save_base_vector(df, PATH_DATA_VECTEUR_FILE)
    else:
        save_base_vector(df, PATH_DATA_VECTEUR + '/' + name_database)


def code_to_name_produit(liste_id):
    return [DF_PRODUIT_TRAIN.loc[id, 'product_name'] for id in liste_id]


def test_performance_cnn(nb_images_test=5, verbose=False):
    """[summary]

    Args:
        nb_images_test (int, optional): [description]. Defaults to 5.
    """
    if verbose:
        nb_image_fait = 0
        print("=========================")
        print("Start create new dataset")
        tps1 = time.time()
    images_a_tester = os.listdir(PATH_DATA_CNN_TEST)
    liste_images = [PATH_DATA_CNN_TEST+'/'+image for image in images_a_tester]
    images_a_tester = [image.replace('.jpg', '') for image in images_a_tester]
    table_index = dict([(i, 0) for i in range(nb_images_test)])
    table_index[-1] = 0
    for path_image, code_origin_image in zip(liste_images, images_a_tester):
        if verbose:
            if (nb_image_fait % 300) == 0:
                tps2 = time.time()
                print(f"nombre traité: {nb_image_fait}")
                print(f"temps d'execution: {tps2 - tps1}")
            nb_image_fait += 1
        try:
            liste_id = image_to_code(path_image, nb_images_test)
        except:
            continue
        try:
            liste_produits = [DF_PRODUIT_TRAIN.loc[id, 'product_name'] for id in liste_id]
        except KeyError as e:
            print(f"code produit not found : {e}")
            continue
        nom_du_produit = DF_PRODUIT_TEST.loc[code_origin_image, 'product_name']
        try:
            table_index[liste_produits.index(nom_du_produit)] += 1
        except ValueError:
            table_index[-1] += 1
    x = [i for i in range(1, nb_images_test + 2)]
    y = [table_index[i] for i in range(nb_images_test)]
    y.append(table_index[-1])
    plt.bar(x, y)
