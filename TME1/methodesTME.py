#############################################################################################
# Imports
#############################################################################################

from ast import If
import numpy as np
import pickle
from collections import Counter
import math
import matplotlib.pyplot as plt
import time
import random

#############################################################################################
# Exercice 1 - Entropie
#############################################################################################

# Méthode permettant de calculer l'entropie d'un vecteur
def entropie(vect) :
    # On compte les probabilités d'apparition dans le vecteur avec un Counter
    count = Counter(vect)
    prob = {key : (nb / len(vect)) for key, nb in count.items()}

    # On retourne l'entropie
    return sum([- prob[i] * math.log(prob[i]) for i in prob])

# Méthode permettant de calculer l'entropie conditionnelle
def entropie_cond(list_vect) :
    # On calcule la longueur totale des vecteurs et la proportion d'éléments
    l = sum([len(vect) for vect in list_vect])
    p = [len(vect) / l for vect in list_vect]

    # On retourne l'entropie conditionelle
    return sum(p[i] * entropie(list_vect[i]) for i in range(len(list_vect)))

#############################################################################################
# Chargement des données
#############################################################################################

# Data : tableau (films, features), id2titles : dictionnaire id -> titre,
# fields : id feature -> nom
[data, id2titles, fields]= pickle.load(open("imdb_extrait.pkl", "rb"))
# La derniere colonne est le vote
datax = data [:,:32]
datay = np.array([1 if x [33] > 6.5 else -1 for x in data ])

#############################################################################################
# Quelques expériences préliminaires
#############################################################################################

from sklearn.tree import export_graphviz
from sklearn.tree import DecisionTreeClassifier as DTree
import pydotplus
# L'affichage des pdf grâce à cette source et la librairie plot_tree
from sklearn.tree import plot_tree

id2genre = [x[1] for x in sorted(fields.items())[: -2]]
dt = DTree()
dt.max_depth = 5 # on fixe la taille max de l’arbre a 5
dt.min_samples_split = 2 # nombre minimum d’exemples pour spliter un noeud
dt.fit(datax, datay)
dt.predict(datax[:5,:])
# print(dt.score(datax, datay))
# utiliser http://www.webgraphviz.com/ par exemple ou https://dreampuf.github.io/Graphvizexport_graphviz ( dt , out_file ="/tmp / tree . dot ", feature_names = id2genre )
# ou avec pydotplus
# tdot = export_graphviz(dt, feature_names = id2genre)
# pydotplus.graph_from_dot_data(tdot).write_pdf("tree.pdf")

# Méthode permettant la création d'un arbre de décision de profondeur précise
# avec un apprentissage sur les données passées en paramètres
def treeMaker(maxDepth, X, Y, doSave=True, doPlot=False) :
    # On crée l'arbre avec une profondeur précise
    dt = DTree(max_depth = maxDepth, min_samples_split = 2)

    # On entraine l'arbre et on le retourne
    dt.fit(X, Y)

    # Affichage de la précision de l'arbre
    print("Précision de l'arbre de décision avec une profondeur maximale de", maxDepth, ":", dt.score(X, Y))

    # Sauvegarde de l'arbre en fichier pdf
    tdot = export_graphviz(dt, feature_names = id2genre)
    fileN = "DTreeImages/tree_max_depth_" + str(maxDepth) + ".pdf"
    pydotplus.graph_from_dot_data(tdot).write_pdf(fileN)

    # Affichage de l'arbre
    if (doPlot) :
        plt.figure(figsize=(maxDepth * 5, maxDepth * 5))
        plot_tree(dt, feature_names = list(fields.values())[:32], filled=True)
        plt.show()

#############################################################################################
# Partie EXEC - Tests
#############################################################################################

# Variables de contrôle des tests
t1 = 0

# Test de l'entropie
if (t1 == 1) :
    vecTest1 = [random.randint(0, 10) for _ in range(5)]
    vecTest2 = [1, 1, 1, 1, 1]

    print("Vecteur :", vecTest1, "\tEntropie :", entropie(vecTest1))
    print("Vecteur :", vecTest2, "\tEntropie :", entropie(vecTest2))