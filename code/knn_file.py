# ========================================
# IMPORT
# ========================================
import pandas as pd
from file_variable import PATH_DATA_VECTEUR_FILE
from sklearn.neighbors import NearestNeighbors
from heapq import nsmallest
# =======================================
# FONCTIONS
# =======================================


# TODO changer construction des KNN et CNN en classe d'objet


# class knn, permet d'import un objet contenant les méthodes lié a un knn
# utilisable pour des vectors d'images

# class knn, permet d'import un objet contenant les méthodes lié a un knn
# utilisable pour des vectors d'images

class class_knn:

    def __init__(self, vecteur_image='', nb_voisins=5):
        self.nb_voisins = nb_voisins
        self.vecteur = vecteur_image
        self.df_vecteur = ''

    def calcul_distance(self,  vecteur1):
        """
        take the vector of the image we are looking for
        take our dataframe of vector directly calculates the distances
        Var :
        vecteur1 : vector of the image we are playing with
        Args:
            code : we are soustracting  also the code in (self.df_vecteur -
            vecteur1)**2, it will also affect code
            So we stock it and replace it
        """
        code = self.df_vecteur['code']
        self.df_vecteur.drop(columns=['code'], inplace=True)
        try:
            self.df_vecteur.drop(columns=['Unnamed: 0'], inplace=True)
            self.df_vecteur.drop(columns=['distance'], inplace=True)
        except:
            pass
        self.df_vecteur = (self.df_vecteur - vecteur1)**2
        self.df_vecteur["distance"] = self.df_vecteur.apply(lambda x: x.sum(),
                                                            axis=1,  raw=True)
        self.df_vecteur["code"] = code

    def calcul_knn(self):
        """
        recuperer les n plus petites distances dans le df
        et renvoies les n codes associé au n plus petites distances
        return liste de 'code'
        """
        min_distance = nsmallest(self.nb_voisins, self.df_vecteur['distance'])
        return self.df_vecteur['code'].loc[self.df_vecteur['distance']
                                           .isin(min_distance)]

    def find_similar_vector_id(self, vecteur, nb_id):
        """returns a list of id's corresponding to the nb_id of similar vectors
        Args:
            vecteur (numpy Array): vector with one line, result of the CNN
            model
            nb_id (int): desired id number, error if nb_id = 0
            liste_tuples_id_vecteurs (tuple) : nom de la photo et son vecteur
            modele (objet) : modèle utilisé
        Returns:
            list(str): list of id image
        """
        self.calcul_distance(vecteur)
        self.vecteur = vecteur
        self.nb_voisins = nb_id
        return self.calcul_knn()

    def save_model(self, model):
        """save in a file the name of the model and the parameters after training

        Args:
            model (Object): model object
        """
        pass

    def charge_database(self, name_database_vector):
        # TODO a faire plus tard pour trouver le chemin de cette database
        self.df_vecteur = pd.read_csv(PATH_DATA_VECTEUR_FILE,  dtype={'code': str})
        pass

    def charge_model(self, PATH_file_model='', name_model=''):
        """loads the model with these parameters into a file

        Args:
            PATH_file_model (string): path where model is save

        Returns:
            Object: model object
        """
        # DF_VECTOR = pd.read_csv(PATH_DATA_VECTEUR_FILE)
        MODEL_CNN = NearestNeighbors(n_clusters=2, random_state=0)
        return MODEL_CNN
