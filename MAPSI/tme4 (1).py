#Tantouch Maher (groupe BMC-BIM)

import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
from IPython.display import Image

import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl

data = pkl.load( open('faithful.pkl', 'rb'))
X = data["X"] 


def normale_bidim(x: np.array, mu: np.array, Sig: np.array):
    # on calcule la différence entre x et mu
    diff = x - mu
    
    # on inverse la matrice Sigma
    Sig_inv = np.linalg.inv(Sig)
    
    # on calcul la forme quadratique
    quadratic_form = diff.T @ Sig_inv @ diff
    
    # on calcul le déterminant de Sigma
    det_Sig = np.linalg.det(Sig)
    
    # on calcul la vraisemblance
    N = len(x)  # Dimension
    likelihood = (1 / ((2 * np.pi) ** (N / 2) * np.sqrt(det_Sig))) * np.exp(-0.5 * quadratic_form)
    
    return likelihood


#def estimation_nuage_haut_gauches(): --> cette version de l fonction indique en sortie des NaN (valeurs invalides) ; 
    # Chargement des données réelles
    #data = pkl.load(open('faithful.pkl', 'rb'))
    #X = data["X"]

    # Estimation de la moyenne
    #mu4 = np.array([4.25, 80.0])  # À ajuster selon vos besoins si nécessaire

    # Estimation des écarts-types
    #std_dev_1 = (np.percentile(X[:, 0], 66.67) - np.percentile(X[:, 0], 33.33)) / 2
    #std_dev_2 = (np.percentile(X[:, 1], 66.67) - np.percentile(X[:, 1], 33.33)) / 2

    # Estimation de la covariance
    #cov = np.cov(X, rowvar=False)

    # Création de la matrice de covariance
    #Sig4 = np.array([[std_dev_1**2, cov[0, 1]], 
                     #[cov[0, 1], std_dev_2**2]])

    #return mu4, Sig4
    

def estimation_nuage_haut_gauche():
    # recuperation des données réelles
    data = pkl.load(open('faithful.pkl', 'rb'))
    X = data["X"]

    # étape ajoutée : on vérifie d'abord les données
    if np.isnan(X).any() or np.isinf(X).any():
        raise ValueError("Les données contiennent des valeurs NaN ou infinies.")

    # éstimation de la moyenne
    mu4 = np.array([4.25, 80.0])  # À ajuster selon vos besoins si nécessaire

    # éstimation des écarts-types
    std_dev_1 = np.std(X[:, 0])
    std_dev_2 = np.std(X[:, 1])

    # on check les écarts-types
    if std_dev_1 <= 0 or std_dev_2 <= 0:
        raise ValueError("Erreur : Écart-type négatif ou nul.")

    # on estime la covariance
    cov = np.cov(X, rowvar=False)

    # on fait la matrice de covariance
    Sig4 = np.array([[std_dev_1**2, cov[0, 1]], 
                     [cov[0, 1], std_dev_2**2]])

    return mu4, Sig4



def init(X):
    # on initialise pi
    pi = np.array([0.5, 0.5])
    
    # moyenne
    mu0 = np.mean(X, axis=0)
    
    # on initialise mu
    mu1 = mu0 + 1  # ajout de 1 sur toutes les dimensions
    mu2 = mu0 - 1  # soustraction de 1 sur toutes les dimensions
    mu = np.array([mu1, mu2])
    
    # on calcul la matrice de covariance de l'ensemble des données
    cov = np.cov(X, rowvar=False)
    
    # on initialise Sigma
    Sig = np.array([cov, cov])  # Deux matrices de covariance identiques

    return pi, mu, Sig



def normale_bidim2(X, mu, Sig):
    # on recalcule la dimension de X
    n = X.shape[0]
    diff = X - mu  # pour la différence entre chaque point et la moyenne
    Sig_inv = np.linalg.inv(Sig)  # on fait l'inverse de la matrice de covariance
    
    #calcule de la forme quadratique
    quadratic_form = np.einsum('ij,jk->i', diff, Sig_inv @ diff.T)  # Produit matriciel
    # calcule de la densité de probabilité
    coeff = (1 / (2 * np.pi * np.sqrt(np.linalg.det(Sig))))
    return coeff * np.exp(-0.5 * quadratic_form)

def Q_i(X, pi, mu, Sig):
    # initialisation de la liste pour stocker les probabilités Q
    Q = []
    
    # on calcul le dénominateur pour chaque observation
    den = np.sum([normale_bidim2(X, mu[j], Sig[j]) * pi[j] for j in range(len(mu))], axis=0)
    
    # on calcul le numérateur et les probabilités Q pour chaque classe
    for i in range(len(mu)):
        num = normale_bidim2(X, mu[i], Sig[i]) * pi[i]
        Q.append(num / den)
    
    # on retourne le tableau des probabilités Q sous forme de tableau NumPy
    return np.array(Q)


import numpy as np

def update_param(X, q, pi, mu, Sig):
    # nombre d'observations
    n = X.shape[0]
    
    # calcul des nouvelles probabilités a priori
    sum_q0 = np.sum(q[0])
    sum_q1 = np.sum(q[1])
    
    pi_u = np.array([sum_q0 / (sum_q0 + sum_q1), sum_q1 / (sum_q0 + sum_q1)])
    
    # calcul des nouvelles moyennes
    mu_u = np.zeros_like(mu)
    for i in range(len(mu)):
        mu_u[i] = np.sum(q[i][:, np.newaxis] * X, axis=0) / np.sum(q[i])
    
    # calcul des nouvelles matrices de covariance
    Sig_u = np.zeros_like(Sig)
    for i in range(len(mu)):
        diff = X - mu_u[i]  # différence entre les observations et la nouvelle moyenne
        Sig_u[i] = (q[i][:, np.newaxis] * diff).T @ diff / np.sum(q[i])
    
    return pi_u, mu_u, Sig_u

def EM(X, initFunc=None, nIterMax=100, saveParam=None):
    # initialisation des paramètres
    if initFunc is None:
        initFunc = init  # utilisez votre fonction d'initialisation ici

    pi, mu, Sig = initFunc(X)

    # itération jusqu'à convergence ou jusqu'à atteindre le nombre maximum d'itérations
    for nIter in range(nIterMax):
        # étape E : calcul des q
        q = Q_i(X, pi, mu, Sig)

        # étape M : mise à jour des paramètres
        pi_u, mu_u, Sig_u = update_param(X, q, pi, mu, Sig)

        # vérification de la convergence sur mu
        if np.allclose(mu, mu_u, atol=1e-5):
            break

        # mise à jour des paramètres pour la prochaine itération
        pi, mu, Sig = pi_u, mu_u, Sig_u

    return nIter + 1, pi, mu, Sig















