#################################################################################################
# Imports
#################################################################################################

import numpy as np
import matplotlib.pyplot as plt

from mltools import plot_data, plot_frontiere, make_grid, gen_arti

#################################################################################################
# MEAN SQUARE ERROR
#################################################################################################

def mse(w,x,y):
    # On fait attention à ce que les données soit dans la bonne dimension
    y = y.reshape(-1, 1)
    w = w.reshape(-1, 1)
    # Autant d'exemples que de labels, autant d'attributs que de poids dans w
    x = x.reshape(y.shape[0], w.shape[0]) 

    # On retourne le coût aux moindres carrés
    return ((np.dot(x, w) - y) ** 2)

def mse_grad(w,x,y):
    # On fait attention à ce que les données soit dans la bonne dimension
    y = y.reshape(-1, 1)
    w = w.reshape(-1, 1)
    # Autant d'exemples que de labels, autant d'attributs que de poids dans w
    x = x.reshape(y.shape[0], w.shape[0]) 
    
    # On retourne le gradient
    return (-2) * x * (y - np.dot(x,w))

#################################################################################################
# LOGISTIC REGRESSION
#################################################################################################

def reglog(w,x,y):
    # On fait attention à ce que les données soit dans la bonne dimension
    y = y.reshape(-1, 1)
    w = w.reshape(-1, 1)
    # Autant d'exemples que de labels, autant d'attributs que de poids dans w
    x = x.reshape(y.shape[0], w.shape[0]) 
    
    # On retourne le coût de la reg. logistique
    return np.log(1 + np.exp((-y) * np.dot(x,w)))

def reglog_grad(w,x,y):
    # On fait attention à ce que les données soit dans la bonne dimension
    y = y.reshape(-1, 1)
    w = w.reshape(-1, 1)
    # Autant d'exemples que de labels, autant d'attributs que de poids dans w
    x = x.reshape(y.shape[0], w.shape[0]) 
    
    # On retourne le gradient
    return ((- y) * x) / (1 + np.exp(y * np.dot(x,w)))

#################################################################################################
# Méthode supplémentaire de vérification
#################################################################################################

def check_fonctions():
    ## On fixe la seed de l'aléatoire pour vérifier les fonctions
    np.random.seed(0)
    datax, datay = gen_arti(epsilon=0.1)
    wrandom = np.random.randn(datax.shape[1],1)
    assert(np.isclose(mse(wrandom,datax,datay).mean(),0.54731,rtol=1e-4))
    assert(np.isclose(reglog(wrandom,datax,datay).mean(), 0.57053,rtol=1e-4))
    assert(np.isclose(mse_grad(wrandom,datax,datay).mean(),-1.43120,rtol=1e-4))
    assert(np.isclose(reglog_grad(wrandom,datax,datay).mean(),-0.42714,rtol=1e-4))
    np.random.seed()

# On pourrait se servir de la méthode de génération fournie dans mltools mais on préfère générer des
# ensembles totalement aléatoires
# Méthode de test d'exactitude
def grad_check(f, f_grad, N=100, eps=1E-2) :
    # Tirage aléatoire des exemples et de w
    x = np.array([np.random.randint(0,100) for i in range(N)]).reshape(-1,1)
    w = np.array([np.random.randint(1,10)])
    y = np.array(np.random.choice([-1,1]))

    # Variable conservant la différence moyenne des gradients
    diffGrad = 0.

    # On sauvegarde même les valeurs calculées
    logGrad = []

    # On parcourt les exemples
    for i in range(N) :
        # On calcule les gradients
        grad = f_grad(w + eps, x[i], y)
        gradT = ((f(w + eps, x[i], y)) - (f(w,x[i],y))) / eps

        # On calcule la différence et on l'ajoute à diffGrad
        diffGrad += np.abs(grad - gradT)

        # On sauvegarde les valeurs
        logGrad.append((grad[0][0], gradT[0][0]))
        

    # On retourne la différence et le log
    return (diffGrad / N), logGrad

#################################################################################################
# Descente de gradient
#################################################################################################

# Descente de gradient batch
def descente_gradient_batch(datax, datay, f_loss, f_grad, eps, maxIter=100, plot=False) :
    # On crée le w initial et les listes de sauvegardes des w et de la fonction de coût
    w = np.random.rand(1, len(datax[0]))
    logW = [w[0].tolist()]
    logF = [np.mean(f_loss(w, datax, datay))]

    # On itère sur le nombre maximal d'itérations
    for iter in range(maxIter) :
        d = np.mean(f_grad(w, datax, datay), axis=0)
        w -= (eps * d)

        # On sauvegarde les valeurs
        logW.append(w[0].tolist())
        logF.append(np.mean(f_loss(w, datax, datay)))

    # Affichage si demandé
    if (plot) :
        # Affichage de l'évolution de la perte
        plt.figure()
        plt.title("Evolution de la loss en fonction du nombre d'itérations en mode batch\n(eps = " + str(eps) + " , maxIter = " + str(maxIter) + ")")
        plt.ylabel("Loss")
        plt.xlabel("Itération")
        plt.plot([(i + 1) for i in range(maxIter + 1)], logF,'r')
        plt.show()

    # On retourne w et les listes
    return w, logW, logF

# MODIFICATION DANS LE CADRE DU TME 4 - MAXITER = EPOCHS
# Descente de gradient stochastique
def descente_gradient_stoch(datax, datay, f_loss, f_grad, eps, maxIter=100, plot=False) :
    # On crée le w initial et les listes de sauvegardes des w et de la fonction de coût
    w = np.random.rand(1, len(datax[0]))
    logW = [w[0].tolist()]
    logF = [np.mean(f_loss(w, datax, datay))]

    # On itère sur le nombre maximal d'itérations
    for iter in range(maxIter * len(datax)) :
        index = np.random.randint(0, len(datax) - 1)
        w -= (eps * f_grad(w, datax[index], datay[index]))

        # On sauvegarde les valeurs
        logW.append(w[0].tolist())
        logF.append(np.mean(f_loss(w, datax, datay)))

    # Affichage si demandé
    if (plot) :
        # Affichage de l'évolution de la perte
        plt.figure()
        plt.title("Evolution de la loss en fonction du nombre d'itérations en mode stochastique\n(eps = " + str(eps) + " , maxIter = " + str(maxIter) + ")")
        plt.ylabel("Loss")
        plt.xlabel("Itération")
        plt.plot([(i + 1) for i in range(maxIter + 1)], logF,'r')
        plt.show()

    # On retourne w et les listes
    return w, logW, logF

# MODIFICATION DANS LE CADRE DU TME 4 - MAXITER = EPOCHS
# Descente de gradient mini-batch
def descente_gradient_mini(datax, datay, f_loss, f_grad, eps, part,  maxIter=100, plot=False) :
    # On crée le w initial et les listes de sauvegardes des w et de la fonction de coût
    w = np.random.rand(1, len(datax[0]))
    logW = [w[0].tolist()]
    logF = [np.mean(f_loss(w, datax, datay))]

    # On itère sur le nombre maximal d'itérations
    for iter in range(maxIter * ((len(datay) / part))) :
        ind = [np.random.randint(0, len(datax) - 1) for i in range(part)] # Indices des exemples à utiliser
        d = np.mean([f_grad(w, datax[i], datay[i]) for i in ind])
        w -= (eps * d)

        # On sauvegarde les valeurs
        logW.append(w[0].tolist())
        logF.append(np.mean(f_loss(w, datax, datay)))

    # Affichage si demandé
    if (plot) :
        # Affichage de l'évolution de la perte
        plt.figure()
        plt.title("Evolution de la loss en fonction du nombre d'itérations en mode mini-batch\n(eps = " + str(eps) + " , maxIter = " + str(maxIter) + ", part = " + str(part) + ")")
        plt.ylabel("Loss")
        plt.xlabel("Itération")
        plt.plot([(i + 1) for i in range(maxIter + 1)], logF,'r')
        plt.show()

    # On retourne w et les listes
    return w, logW, logF

#################################################################################################
# Main
#################################################################################################

if __name__=="__main__":
    ## Tirage d'un jeu de données aléatoire avec un bruit de 0.1
    datax, datay = gen_arti(epsilon=0.1)
    ## Fabrication d'une grille de discrétisation pour la visualisation de la fonction de coût
    grid, x_grid, y_grid = make_grid(xmin=-2, xmax=2, ymin=-2, ymax=2, step=100)
    
    plt.figure()
    ## Visualisation des données et de la frontière de décision pour un vecteur de poids w
    w  = np.random.randn(datax.shape[1],1)
    plot_frontiere(datax,lambda x : np.sign(x.dot(w)),step=100)
    plot_data(datax,datay)

    ## Visualisation de la fonction de coût en 2D
    plt.figure()
    plt.contourf(x_grid,y_grid,np.array([mse(w,datax,datay).mean() for w in grid]).reshape(x_grid.shape),levels=20)
    
