import numpy as np
import math

import matplotlib.pyplot as plt

def bernoulli (p):
    t = np.random.random()
    if t<p :
        return 1
    else:
        return 0


def binomiale(n,p):
    return np.array([bernoulli(p) for i in range (n)]).sum()

#commentaire : pourquoi je n'utilise pas la fonction print()

#def galton (l, n, p): à voir

#def histo.galton(): à voir

def normale(k, sigma):
    assert k % 2 == 1, "k devrait être une valeur impair"
    espace = (4 * sigma) / k
    abscisse = np.array([-2 * sigma + i * espace for i in range(k)])
    print(abscisse)
    return (1 / sigma * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((abscisse / sigma) ** 2))


def proba_affine(k, slop):
    if k % 2 == 0:
        raise ValueError('la valeur de k devrait être impair')
    if abs(slop) > 2. / (k * k):
        raise ValueError('la pente est trop grande : pente maximale possible = ' + str(2. / (k * k)))

    x = np.arange(0, k)
    Yi = lambda i: 1/k + (i - ((k - 1) / 2)) * slop
    return Yi(x)



def Pxy(x, y):
    return np.outer(x, y)

#def calcYZ(a):
    #return np.sum(a)


import numpy as np

import numpy as np


def calcYZ(probabilite): 
    """
    Ici on calcule P(Y, Z) en sommant toutes les dimensions sauf celles de Y et Z.

    En sortie: numpy.ndarray: Tableau de probabilités P(Y, Z).
    """
    # Déterminer les dimensions Y et Z : on suppose que ce sont les deux dernières dimensions ici
    return np.sum(probabilite, axis=tuple(range(probabilite.ndim - 2)))  # Somme sur toutes les dimensions sauf les deux dernières




#def calcXTcondYZ():
import numpy as np

def calcXTcondYZ(P_XYZT):
    """
    Ici on calcule P(X, T | Y, Z) à partir de P(X, Y, Z, T).

    En sortie : numpy.ndarray: Tableau de probabilités P(X, T | Y, Z).
    """
    # Calcul de P(Y, Z) cf 
    P_YZ = np.sum(P_XYZT, axis=(0, 3))  # ou bien on utilise notre fonction précédente: P_YZ = calcYZ(P_XYZT)
    
    # Assurer que P(Y, Z) n'est pas zéro pour éviter la division par zéro
    if np.any(P_YZ == 0):
        raise ValueError("P(Y, Z) contient des zéros, impossible de calculer P(X, T | Y, Z)")

    # Calculer P(X, T | Y, Z)
    P_XTcondYZ = P_XYZT / P_YZ[np.newaxis, :, :, np.newaxis]  # Utilisation de broadcasting

    return P_XTcondYZ


#def calcX_etcondYZ():
def calcX_et_TcondYZ(P_XTYZ):
    """
    ici on calcule P(X | Y, Z) et P(T | Y, Z) à partir de P(X, T, Y, Z).

    en sortie : Tableaux de probabilités P(X | Y, Z) et P(T | Y, Z).
    """
    P_YZ = calcYZ(P_XTYZ)   # calcul de P(Y, Z) en utilisant la fonction calcYZ

    # vérification pour ne pas diviser par des zéros
    if np.any(P_YZ == 0):
        raise ValueError("P(Y, Z) contient des zéros, impossible de calculer P(X | Y, Z) et P(T | Y, Z)")

    P_XcondYZ = np.sum(P_XTYZ, axis=1) / P_YZ[np.newaxis, :]  # on calcul ici P(X | Y, Z) en sommant P(X, T | Y, Z) (on omme sur la dimension T)


    P_TcondYZ = np.sum(P_XTYZ, axis=0) / P_YZ[np.newaxis, :]  # On calcul ici P(T | Y, Z) en utilisant P(X, T | Y, Z) on somme sur la dimension X

    return P_XcondYZ, P_TcondYZ



def testXTindepCondYZ(P_XYZT, epsilon=1e-10): #epsilon (float): sorte de tolérance pour vérifier l'égalité numérique.
    """
    ici on cherche à vérifier si X et T sont indépendantes conditionnellement à Y et Z dans la distribution P(X, Y, Z, T).
    En sortie : True si X et T sont indépendantes conditionnellement à Y et Z, False sinon.
    """
    
    
    P_XTcondYZ = calcXTcondYZ(P_XYZT) # On calcule ici P(X, T | Y, Z)

    P_XcondYZ, P_TcondYZ = calcX_et_TcondYZ(P_XYZT) # Ici on calcule P(X | Y, Z) et P(T | Y, Z)

    return np.allclose(P_XTcondYZ, P_XcondYZ * P_TcondYZ, atol=epsilon) # ici on vérifie l'égalité à l'intérieur de la tolérance epsilon


#def testXindepYZ():
import numpy as np


def calcYZ(probabilite):
    """
    Calcule P(Y, Z) en sommant toutes les dimensions sauf celles de Y et Z.
    """
    return np.sum(probabilite, axis=tuple(range(probabilite.ndim - 2)))  # Somme sur toutes les dimensions sauf les deux dernières

import numpy as np

import numpy as np

import numpy as np

import numpy as np

def testXindepYZ(P_XYZT, epsilon=1e-10):
    """
    ici on vérifie si X est indépendante de (Y, Z) dans la distribution P(X, Y, Z, T).

    en sortie : Vérification si le tableau est bon (P_X) ; et True si X est indépendante de (Y, Z), False sinon.
    """
    # Étape 1 : Calculer P(X, Y, Z) en sommant sur T
    P_XYZ = P_XYZT.sum(axis=3)  # Somme sur T, dimensions (X, Y, Z)

    # Étape 2 : Calculer P(X)
    P_X = P_XYZ.sum(axis=2).sum(axis=1)  # Somme sur Z puis Y
    print(P_X)

    # Étape 3 : Calculer P(Y, Z)
    P_YZ = P_XYZ.sum(axis=0)  # Somme sur X pour obtenir P(Y, Z)

    # Étape 4 : Vérifier l'indépendance
    P_X_YZ = np.zeros_like(P_XYZ)

    for x in range(P_XYZ.shape[0]):
        P_X_YZ[x] = P_X[x] * P_YZ

    # Comparer P(X, Y, Z) avec P(X) * P(Y, Z)
    return np.allclose(P_XYZ, P_X_YZ, atol=epsilon)


#def conditional_indep():
import numpy as np

import numpy as np

import numpy as np

import numpy as np

def conditional_indep(P, X, Y, Zs=[], epsilon=1e-10):
    """
    ici on vérifie si X et Y sont indépendants conditionnellement à Zs dans le potentiel P.

    en sortie : True si X et Y sont indépendants conditionnellement à Zs, False sinon
    """
    
    # on vérifie ici si aucune variable conditionnelle n'est fournie
    if not Zs:  # pour dire Zs est vide
        
        return (P.margSumIn([X, Y]) - P.margSumIn(X) * P.margSumIn(Y)).abs().max() < epsilon # vérification de l'indépendance sans conditions

    # On calcul ensuite les probabilités conditionnelles avec Zs
    P_XYZs = P.margSumIn([X, Y] + Zs)
    P_XY_Zs = P_XYZs / P.margSumIn(Zs)
    P_X_Zs = P.margSumIn([X] + Zs) / P.margSumIn(Zs)
    P_Y_Zs = P.margSumIn([Y] + Zs) / P.margSumIn(Zs)

    # Enfin, on vérifie l'indépendance conditionnelle
    return (P_XY_Zs - P_X_Zs * P_Y_Zs).abs().max() < epsilon


def compact_conditional_proba(P, X):
    """
    ici on veut avoir la probabilité conditionnelle compacte P(X | K) à partir du potentiel P.
    en sortie : P(X | K), la probabilité conditionnelle compacte
    """
    
   
    K = list(P.names).copy()  # Création d'une copie des noms de variables dans P, en excluant X
    K.remove(X)

    # Évaluation de l'indépendance conditionnelle et ajuster K
    for var in K[:]:  # pour faire un copie de K pour éviter les problèmes de modification éventuelles
        K_priv_var = K.copy()
        K_priv_var.remove(var)

        
        if conditional_indep(P, var, X, K_priv_var): # Vérification de l'indépendance conditionnelle
            K.remove(var)

    
    P_XK = P.margSumIn(K + [X]) # on calcule la probabilité conditionnelle P(X, K)
    
    # si K non vide on normalise
    if not K:
        P_X_K = P_XK
    else:
        P_X_K = P_XK / P.margSumIn(K)

    
    return P_X_K.putFirst(X) # on placera X en premier dans le résultat


def create_bayesian_network(P, show_names=True):
    """
    on souhaite créer un réseau bayésien à partir d'un potentiel P.
    en sortie : liste de potentiels représentant le réseau bayésien
    """
    
    
    network = [] # on initialise une liste pour stocker les potentiels
    
    variables = list(P.names) # Obtention de la liste des noms de variables dans P

    # on traite ici les variables en ordre inverse
    for variable in reversed(variables):
    
        Q = compact_conditional_proba(P, variable)# calcul de la probabilité conditionnelle compacte pour la variable actuelle
        
        
        if show_names: # affichage des noms des variables si show_names est activé
            print(Q.names)
        
        # 
        network.append(Q)
        
        # Éliminer la variable actuelle de P
        P = P.margSumOut(variable)

    return network


def calcNbParams(Pjointe):
    """
    Calcule le nombre de paramètres dans la loi jointe et dans le réseau bayésien créé à partir de cette loi.

    Parameters:
    - Pjointe : Potential représentant la loi jointe

    Returns:
    - tuple : (nombre de paramètres dans la loi jointe, nombre de paramètres dans le réseau bayésien)
    """
    # Nombre de paramètres dans la loi jointe
    taille_jointe = Pjointe.domainSize()

    # Créer le réseau bayésien à partir de la loi jointe
    réseau_bayésien = create_bayesian_network(Pjointe, show_names=False)

    # Calculer le nombre total de paramètres dans le réseau bayésien
    taille_rb = sum(potential.domainSize() for potential in réseau_bayésien)

    return taille_jointe, taille_rb



    

