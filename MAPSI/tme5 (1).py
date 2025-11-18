#Maher Tantouch (Grpe BMC-BIM)

import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
from IPython.display import Image

import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
    
import numpy as np
import utils
import tme5

import numpy as np
import scipy.stats as stats


def sufficient_statistics(data, dico, x, y, z):
    # on créé un tableau de contingence T_{X,Y|Z}
    T = utils.create_contingency_table(data, dico, x, y, z)

    # on doit aussi initialiser les paramètres qui nous interessent
    chi2 = 0
    total_degrees_of_freedom = 0
    valid_z_count = 0
    
    # parcours du tableau de contingence
    for i in range(len(T)):
        N_total = T[i][0]  # --> c'est ici le nombre total d'enregistrements pour la valeur de Z
        N_xy = T[i][1]  # pour le tableau des occurrences de (X,Y) pour la valeur de Z

        if N_total > 0:
            # ensuite on calcul N_xz et N_yz
            N_xz = np.sum(N_xy, axis=1)  # Somme sur les lignes pour N_xz
            N_yz = np.sum(N_xy, axis=0)  # Somme sur les colonnes pour N_yz
            
            # calcul du chi2 pour cette valeur de Z
            for x_val in range(len(N_xz)):
                for y_val in range(len(N_yz)):
                    O = N_xy[x_val, y_val]  # Observé
                    E = (N_xz[x_val] * N_yz[y_val]) / N_total if N_total > 0 else 0  # Attendu

                    if E > 0:  # on ne divise pas par 0
                        chi2 += (O - E) ** 2 / E
            
            valid_z_count += 1  # Compte des Z valides

    # le degré de liberté : (|X| - 1) * (|Y| - 1) * |{Z : N_z ≠ 0}|
    degrees_of_freedom = (len(dico[x]) - 1) * (len(dico[y]) - 1) * valid_z_count

    return chi2, degrees_of_freedom

import scipy.stats as stats

def indep_score(data, dico, x, y, z):
    # on débute par determiner le nombre de valeurs uniques pour X, Y, Z
    num_X = len(dico[x])
    num_Y = len(dico[y])
    num_Z = len(z) if z else 1  # Si Z est vide, considérer 1

    # on calcul d_min
    d_min = 5 * num_X * num_Y * num_Z

    # vérification de si le nombre d'enregistrements est suffisant
    if data.shape[1] < d_min:
        return (-1, 1)  # Indépendance si le seuil n'est pas respecté

    # on calcule ensuite le χ² et les degrés de liberté
    chi2, dof = sufficient_statistics(data, dico, x, y, z)

    # calcul de la p-value
    p_value = stats.chi2.sf(chi2, dof)

    return p_value, dof



def best_candidate(data, dico, x, z, alpha):
    best_y = None
    min_p_value = float('inf')
    
    # parcours des colonnes à gauche de X
    for y in range(x):
        #  p-value pour X et Y conditionnellement à Z
        p_value, dof = indep_score(data, dico, x, y, z)
        
        # on check si la p-value est inférieure à alpha
        if p_value < alpha:
            if p_value < min_p_value:
                min_p_value = p_value
                best_y = y  # Enregistrer l'index de Y

    # si  variable Y trouvée, retour de son index dans une liste
    return [best_y] if best_y is not None else []

def create_parents(data, dico, x, alpha):
    z = []  # on initialise l'ensemble des parents
    
    while True:
        candidate = best_candidate(data, dico, x, z, alpha)
        
        if not candidate:  # --> aucun candidat n'est trouvé
            break
        
        # ajout du candidat à z
        z.append(candidate[0])
    
    return z

def learn_BN_structure(data, dico, alpha):
    n = data.shape[0]  # pour le nombre de variables
    bn_struct = [[] for _ in range(n)]  # on initialise la structure du BN
    
    for x in range(n):
        parents = create_parents(data, dico, x, alpha)
        bn_struct[x] = parents  # Ajout des parents associé au nœud x
    
    return bn_struct

def learn_parameters ( bn_struct, ficname ):
    # création du dag correspondant au bn_struct
    graphe = gum.DAG ()
    nodes = [ graphe.addNode () for i in range ( bn_struct.shape[0] ) ]
    for i in range ( bn_struct.shape[0] ):
        for parent in bn_struct[i]:
            graphe.addArc ( nodes[parent], nodes[i] )

    # appel au BNLearner pour apprendre les paramètres
    learner = gum.BNLearner ( ficname )
    learner.useScoreLog2Likelihood ()
    learner.useSmoothingPrior ()
    return learner.learnParameters ( graphe )






