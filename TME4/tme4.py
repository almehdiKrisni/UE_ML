#################################################################################################
# Imports
#################################################################################################

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mltools import plot_data, plot_frontiere, make_grid, gen_arti
from tme3_for_tme4 import *

#################################################################################################
# Perceptron
#################################################################################################

def perceptron_loss(w,x,y) :
    # Comme dans le TME3
    # On fait attention à ce que les données soit dans la bonne dimension
    y = y.reshape(-1, 1)
    w = w.reshape(-1, 1)
    # Autant d'exemples que de labels, autant d'attributs que de poids dans w
    x = x.reshape(y.shape[0], w.shape[0])

    # On retourne la loss du perceptron
    return np.maximum(0, - (y * np.dot(x,w)))

def perceptron_grad(w,x,y) :
    # On fait attention à ce que les données soit dans la bonne dimension
    y = y.reshape(-1, 1)
    w = w.reshape(-1, 1)
    # Autant d'exemples que de labels, autant d'attributs que de poids dans w
    x = x.reshape(y.shape[0], w.shape[0])

    # Le perceptron ne réalise une correction, donc on cherche les points mal classés
    classif = (y * np.dot(x,w)).flatten()
    indexBadClassif = np.where(classif < 0)

    # On crée le gradient à retourner
    grad = np.zeros(len(x))
    grad[indexBadClassif] = ((-y) * x)[indexBadClassif]
    return grad

#################################################################################################
# IClasse Linéaire
#################################################################################################

class Lineaire(object):
    def __init__(self,loss=perceptron_loss,loss_g=perceptron_grad,max_iter=100,eps=0.01):
        self.max_iter, self.eps = max_iter,eps
        self.w = None
        self.loss,self.loss_g = loss,loss_g
        
    def fit(self,datax,datay,mode="batch",part=10):
        # On sauvegarder les ensembles d'apprentissage
        self.trainx = datax
        self.trainy = datay

        # On retourne la descente de gradient
        if (mode == "batch") :
            self.w, self.wLog, self.fLog = descente_gradient_batch(self.trainx, self.trainy, self.loss, self.loss_g, eps=self.eps, maxIter=self.max_iter)
        
        elif (mode == "stoch") :
            self.w, self.wLog, self.fLog = descente_gradient_stoch(self.trainx, self.trainy, self.loss, self.loss_g, eps=self.eps, maxIter=self.max_iter)

        elif (mode == "mini") :
            self.w, self.wLog, self.fLog = descente_gradient_mini(self.trainx, self.trainy, self.loss, self.loss_g, eps=self.eps, part=part, maxIter=self.max_iter)

        # Aucun mode ne correspond
        else :
            print("Le paramètre 'mode' n'a pas été reconnu. Veuillez le modifier.")

    def predict(self,datax):
        # On sauvegarder l'ensemble de test
        self.testx = datax

        # On retourne la prédiction avec le signe du produit scalaire
        return np.sign(np.dot(self.testx, self.w.reshape(-1,1)))

    def score(self,datax,datay):
        # On reourne le score de classification sur les données passées en paramètres
        # On sauvegarde les ensembles
        self.testx = datax
        self.testy = datay

        # On prédit les labels de datax et on retourne la précision
        pred = self.predict(self.testx)
        return np.mean([1 if (self.testy[i] == pred[i]) else 0 for i in range(len(self.testy))])

#################################################################################################
# Gestion données USPS
#################################################################################################

def load_usps(fn):
    with open(fn,"r") as f:
        f.readline()
        data = [[float(x) for x in l.split()] for l in f if len(l.split())>2]
    tmp=np.array(data)
    return tmp[:,1:],tmp[:,0].astype(int)

def get_usps(l,datax,datay):
    if type(l)!=list:
        resx = datax[datay==l,:]
        resy = datay[datay==l]
        return resx,resy
    tmp =   list(zip(*[get_usps(i,datax,datay) for i in l]))
    tmpx,tmpy = np.vstack(tmp[0]),np.hstack(tmp[1])
    return tmpx,tmpy

def show_usps(data):
    plt.imshow(data.reshape((16,16)),interpolation="nearest",cmap="gray")

#################################################################################################
# Main
#################################################################################################

if __name__ =="__main__":
    uspsdatatrain = "../data/USPS_train.txt"
    uspsdatatest = "../data/USPS_test.txt"
    alltrainx,alltrainy = load_usps(uspsdatatrain)
    alltestx,alltesty = load_usps(uspsdatatest)
    neg = 5
    pos = 6
    datax,datay = get_usps([neg,pos],alltrainx,alltrainy)
    testx,testy = get_usps([neg,pos],alltestx,alltesty)
