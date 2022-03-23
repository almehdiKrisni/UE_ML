#######################################################################################
# Imports
#######################################################################################

from re import X
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
import pandas as pd

#######################################################################################
# Chargement des données - API Google Places (image et pkl)
#######################################################################################

POI_FILENAME = "data/poi-paris.pkl"
poidata = pickle.load(open(POI_FILENAME, "rb"))
parismap = mpimg.imread('data/paris-48.806-2.23--48.916-2.48.jpg')
## coordonnees GPS de la carte
xmin, xmax = 2.23, 2.48  # coord_x min et max
ymin, ymax = 48.806, 48.916  # coord_y min et max
coords = [xmin, xmax, ymin, ymax]

#######################################################################################
# Classe Density
#######################################################################################

class Density(object):
    def fit(self,data):
        pass
    
    def predict(self,data):
        pass

    def score(self, data, biais=10E-10):
        # On retourne le score du classifieur
        return np.log(self.predict(data) + biais).sum()

#######################################################################################
# Classe Histogramme
#######################################################################################

class Histogramme(Density):
    def __init__(self,steps=10):
        Density.__init__(self)
        self.steps = steps

        # On sauvegarde le pas de discrétisation sur chaque axes (2D)
        self.stepx = ((xmax - xmin) / self.steps)
        self.stepy = ((ymax - ymin) / self.steps)

    def fit(self,data):
        # Utilisation de np.histogramdd et sauvegarde du résultat et de la taille
        hist, _ = np.histogramdd(data, bins=(self.steps, self.steps))
        self.hist = hist
        self.size = len(data)

    def predict(self,data):
        # Prédiction sur des données
        pred = []

        # On parcourt les données que l'on souhaite prédire
        for i in range(len(data)) :
            # On calcule les coordonnées
            cordx = int((data[i][0] - xmin) / self.stepx)
            cordy = int((data[i][1] - ymin) / self.stepy)
            pred.append(self.hist[cordx][cordy])

        # On retourne la prédiction
        return np.array(pred) / ((self.stepx * self.stepy) * self.size)

#######################################################################################
# Classe KernelDensity
#######################################################################################

class KernelDensity(Density):
    def __init__(self,kernel=None,sigma=0.1):
        Density.__init__(self)
        self.kernel = kernel
        self.sigma = sigma

    def fit(self,x):
        # On lie les données à l'histogramme
        self.x = x

    def predict(self,data):
        #A compléter : retourne la densité associée à chaque point de data
        pass

#######################################################################################
# Méthodes complémentaires fournies
#######################################################################################

def get_density2D(f,data,steps=100):
    """ Calcule la densité en chaque case d'une grille steps x steps dont les bornes sont calculées à partir du min/max de data. Renvoie la grille estimée et la discrétisation sur chaque axe.
    """
    xmin, xmax = data[:,0].min(), data[:,0].max()
    ymin, ymax = data[:,1].min(), data[:,1].max()
    xlin,ylin = np.linspace(xmin,xmax,steps),np.linspace(ymin,ymax,steps)
    xx, yy = np.meshgrid(xlin,ylin)
    grid = np.c_[xx.ravel(), yy.ravel()]
    res = f.predict(grid).reshape(steps, steps)
    return res, xlin, ylin

def show_density(f, data, steps=100, log=False):
    """ Dessine la densité f et ses courbes de niveau sur une grille 2D calculée à partir de data, avec un pas de discrétisation de steps. Le paramètre log permet d'afficher la log densité plutôt que la densité brute
    """
    res, xlin, ylin = get_density2D(f, data, steps)
    xx, yy = np.meshgrid(xlin, ylin)
    plt.figure(figsize=(10,10))
    show_img()
    if log:
        res = np.log(res+1e-10)
    plt.scatter(data[:, 0], data[:, 1], alpha=0.8, s=3)
    plt.colorbar()
    plt.contour(xx, yy, res, 20)
    show_img(res)
    

def show_img(img=parismap):
    """ Affiche une matrice ou une image selon les coordonnées de la carte de Paris.
    """
    origin = "lower" if len(img.shape) == 2 else "upper"
    alpha = 0.3 if len(img.shape) == 2 else 1.
    
    plt.imshow(img, extent=coords, aspect=1.5, origin=origin, alpha=alpha)
    ## extent pour controler l'echelle du plan


def load_poi(typepoi,fn=POI_FILENAME):
    """ Dictionaire POI, clé : type de POI, valeur : dictionnaire des POIs de ce type : (id_POI, [coordonnées, note, nom, type, prix])
    
    Liste des POIs : furniture_store, laundry, bakery, cafe, home_goods_store, 
    clothing_store, atm, lodging, night_club, convenience_store, restaurant, bar
    """
    poidata = pickle.load(open(fn, "rb"))
    data = np.array([[v[1][0][1],v[1][0][0]] for v in sorted(poidata[typepoi].items())])
    note = np.array([v[1][1] for v in sorted(poidata[typepoi].items())])
    return data,note

#######################################################################################
# Méthodes créées
#######################################################################################

# Méthode réalisant un affichage de la carte en affichant le type de POI souhaité
def show_poi(typepoi, img=parismap, fn=POI_FILENAME) :
    # Récupération des POI
    geo_mat, notes = load_poi(typepoi, fn=fn)

    # Affichage de l'image et des POI
    print("Carte de Paris avec affichage des " + typepoi)
    show_img(img=img)
    plt.scatter(geo_mat[:,0],geo_mat[:,1],alpha=0.8,s=3)

#######################################################################################
# Affichage des données
#######################################################################################

plt.ion()
# Liste des POIs : furniture_store, laundry, bakery, cafe, home_goods_store, clothing_store, atm, lodging, night_club, convenience_store, restaurant, bar
# La fonction charge la localisation des POIs dans geo_mat et leur note.
geo_mat, notes = load_poi("bar")

# Affiche la carte de Paris
# show_img()

# Affiche les POIs
# plt.scatter(geo_mat[:,0],geo_mat[:,1],alpha=0.8,s=3)