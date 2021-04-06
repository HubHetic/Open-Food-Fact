# ========================================
# IMPORT
# ========================================
import pandas as pd
from sklearn.cluster import KMeans
from file_variable import PATH_DATA_VECTEUR_FILE
from sklearn.neighbors import NearestNeighbors
from heapq import nsmallest
# =======================================
# VARIABLE GLOBAL
# =======================================
MODEL_KNN = ''
DF_VECTOR = ''
# =======================================
# FONCTIONS
# =======================================


# TODO changer construction des KNN et CNN en classe d'objet


#class knn, permet d'import un objet contenant les méthodes lié a un knn utilisable pour des vectors d'images
from heapq import nsmallest
#class knn, permet d'import un objet contenant les méthodes lié a un knn utilisable pour des vectors d'images
class class_knn :
  def __init__(self, nb_voisins, model_knn, vecteur_image, df_vector):
    self.model_knn = model_knn
    self.nb_voisins = nb_voisins
    self.vecteur = vecteur_image
    self.df_vecteur = panda.read_csv(PATH_DATA_VECTEUR)  #a_verif


  def calcul_distance(self,  vecteur1): 
    """take the vector of the image we are looking for
        take our dataframe of vector directly calculates the distances
        

    Var :   
         vecteur1 : vector of the image we are playing with       


    Args:
        code : we are soustracting  also the code in (self.df_vecteur - vecteur1)**2, it will also affect code
    So we stock it and replace it 

    """
    code = self.df_vecteur['code'] #applique le calcul de distance sur code aussi
    self.df_vecteur = (self.df_vecteur - vecteur1)**2
    self.df_vecteur['code'] = code
    self.df_vecteur['distance'] = 0
    for i in self.df_vecteur.columns: 
      if i!= 'code' and i!= 'distance' :
        self.df_vecteur['distance'] = self.df_vecteur['distance'] + self.df_vecteur[i]
    #return 


  def calcul_knn(self):    # no need call liste_tuples_id_vecteurs dans l'appel de la fonction vu que var global
        """recuperer les n plus petites distances dans le df
     et renvoies les n codes associé au n plus petites distances

    
        return liste de 'code'
        """ 
        min_distance = nsmallest(self.nb_voisins , self.df_vecteur['distance'])
        return self.df_vecteur['code'].loc[self.df_vecteur['distance'].isin(min_distance)]




def find_similar_vector(vector):
    """return the vector of the similar image

    Args:
        vector (numpy Array): vector with one line, result of the CNN model

    Returns:
        numpy Array: vector of the similar image in the vector database
    """
    neighboors = MODEL_KNN.kneighbors(vector)
    return neighboors


def find_similar_vector_id(vecteur, nb_id):
    """returns a list of id's corresponding to the nb_id of similar vectors

    Args:
        vecteur (numpy Array): vector with one line, result of the CNN model
        nb_id (int): desired id number, error if nb_id = 0
        liste_tuples_id_vecteurs (tuple) : nom de la photo et son vecteur
        modele (objet) : modèle utilisé

    Returns:
        list(str): list of id image
    """
    return [""]


# Permet de trouver le vecteur le plus proche de celui passé en paramètre
# parmi ceux de la bdd
"""
def calcul_distance(vecteur1, vecteur2):
    return sum((vecteur1 - vecteur2)**2)


def knn(vecteur, liste_tuples_id_vecteurs):
    liste = [calcul_distance(x[1], vecteur[1]) for x in
    liste_tuples_id_vecteurs]
    return liste_tuples_id_vecteurs[liste.index(min(liste))][0]
"""


def test_performance_model(model):
    """displays the performance of the model with different metrics, using graphs

    Args:
        model (model Object): already trained model that you want to test
    """
    pass


def train_model(nb_vecteur, hp_model):
    """train model cnn with its hyperparameters

    Args:
        nb_image (int): number image for train
        hp_model (dico): dictionnaire with hyperparameters

    Returns:
        Object model: model train
    """
    x = DF_VECTOR.drop(['code'])
    MODEL_KNN.fit(x)
    return MODEL_KNN


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


def charge_model(self, PATH_file_model='', name_model=''):
    """loads the model with these parameters into a file

    Args:
        PATH_file_model (string): path where model is save

    Returns:
        Object: model object
    """
    global MODEL_KNN, DF_VECTOR
    DF_VECTOR = pd.read_csv(PATH_DATA_VECTEUR_FILE)
    MODEL_CNN = NearestNeighbors(n_clusters=2, random_state=0)
    return MODEL_CNN


 