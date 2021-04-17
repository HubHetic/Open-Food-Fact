# ========================================
# IMPORT
# ========================================
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
# =======================================
# CLASS
# =======================================


class Class_Knn:
    """knn class : 
            our goal here is to implement a knn on our vectorized image        
    """

    def __init__(self, vecteur_image='', nb_voisins=5):
        self.nb_voisins = nb_voisins
        self.model = ''
        self.df_vecteur = ''

    def train(self):
        """function to train our knn on our df of vector
        """
        df = self.df_vecteur.drop(columns=['code'])
        self.model.fit(df)

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
        vecteur = np.array(vecteur).reshape(1, -1)
        l_index = self.model.kneighbors(vecteur, n_neighbors=nb_id,
                                        return_distance=False)[0]
        return [self.df_vecteur.loc[index, 'code'] for index in l_index]

    def save_model(self, model):
        # TODO save dataframe 
        """save in a file the name of the model and the parameters after training

        Args:
            model (Object): model object
        """
        pass

    def load_knn_df(self, name_database_vector):
        # TODO a faire plus tard pour trouver le chemin de cette database
        self.df_vecteur = pd.read_csv(name_database_vector,  dtype={'code': str})

    def charge_model(self, name_model=''):
        """loads the model with these parameters into a file

        Args:
            PATH_file_model (string): path where model is save

        Returns:
            Object: model object
        """
        self.model = NearestNeighbors(n_neighbors=5)
