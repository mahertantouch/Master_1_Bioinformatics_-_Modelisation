#Maher Tantouch Groupe 4 (BMC-BIM)

import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt


def learnHMM(allX, allS, N, K):
    # tout d'abord on initialise les matrices A et B resp. de transition et d'émission avec des zéros uniquement
    matrice_transition = np.zeros((N, N))  # matrice de transition entre les états
    matrice_emission = np.zeros((N, K))  # matrice de probabilité des émissions pour chaque état

    # on compte les transitions entre les états et les émissions
    for i in range(len(allX) - 1):  # des séquences sauf la dernière observation
        matrice_transition[allS[i], allS[i + 1]] += 1  # transition d'un état vers le suivant
        matrice_emission[allS[i], allX[i]] += 1  # probabilité d'émission de l'observation pour chaque état
    
    # on ajoute séparément l'émission pour la dernière observation de la séquence
    matrice_emission[allS[-1], allX[-1]] += 1

    # on normalise pour obtenir une distribution de probabilités (de sorte à ce que chaque ligne somme à 1) :
    # pour cela on divise chaque ligne par sa somme
    matrice_transition = (matrice_transition.T / matrice_transition.sum(axis=1)).T 
    matrice_emission = (matrice_emission.T / matrice_emission.sum(axis=1)).T  

    return matrice_transition, matrice_emission


#Discussion initialisation avec 1 plutôt que 0 : En initialisant par des 1, on garantit que chaque transition ou observation debute avec une probabilite positive, ce qui rvite les difficultes liees à l'estimation des probabilite de transition ou d'emissions négatives, particulierement si certaines transitions ou observations n'ont pas eu lieu au cours du processus d'entrzinement. Dans le tme6 on a du faire semblant d'observer de chaque type avant même le début du comptage : avec une initialisation de 1 , ceci n'est plus necessaire.

def viterbi(allx, Pi, A, B):
    """
    Parametres
    ----------
    allx : array (T,)
        sequence d'obs.
    Pi: array, (K,)
        distr de probabilité initiale.
    A : array (K, K)
        matrice de transition entre les etats.
    B : array (K, M)
        matrice d'emission des obs par etat.
    
    Returns
    -------
    etats_predits : array (T,)
        seq d'etats caches predite.
    """
    
    T = len(allx)  # = nombre d'observations
    K = len(Pi)    # = nombre d'états
    M = len(B[0])  # = nombre de symboles possibles

    # on initialise les matrices de proba maximales (=matrices delta) et de chemin optimal (psi) - ces derniers indiquant les indices des états à la proba #maximales
    delta = np.zeros((K, T)) 
    psi = np.zeros((K, T), dtype=int)  
    
    # on initialise la première colonne de la mztrice delta
    for k in range(K):
        delta[k, 0] = np.log(Pi[k]) + np.log(B[k, allx[0]])  #état initial
    
    #recursion
    for t in range(1, T):
        for k in range(K):
            #on doit determiner delta[k, t] pour chaque ztat k d'oobservation t
            log_prob = delta[:, t-1] + np.log(A[:, k])  #des etats precedents vers l'etat k
            psi[k, t] = np.argmax(log_prob)  # pour trouver l'état precedent k qui maximise la probabilite
            delta[k, t] = np.max(log_prob) + np.log(B[k, allx[t]])  #on ajoute ensuite la probabilite d'emission
    
    #ztape de terminaison
    etats_predits = np.zeros(T, dtype=int)
    etats_predits[T-1] = np.argmax(delta[:, T-1])  #= etat final a la proba maximale
    
    #retraçage du chemin optimal
    for t in range(T-2, -1, -1):
        etats_predits[t] = psi[etats_predits[t+1], t+1]
    
    return etats_predits



def get_and_show_coding(etat_predits, annotation_test):
    """
    la fonction prend en entree deux seq d'etats (etat_predits=etats predits par l'algorithme & annotation_test= etats reels)

    et renvoie deux tableaux (codants_predits=sequence binaire &  codants_tests où le 1 indique un etat codant et 0 un non codant dans les predictions de l'algo)
    """
    
    #on cinvertit chaque états en "codant" (1 pour codant, 0 pour non-codant)
    codants_predits = np.where(etat_predits == 0, 0, 1) 
    codants_tests = np.where(annotation_test == 0, 0, 1)  

    
    print("les codants predits : ", codants_predits)
    print("les codants réels (annotations) : ", codants_tests)

    return codants_predits, codants_tests

import numpy as np
import matplotlib.pyplot as plt

def create_confusion_matrix(codants_predits, codants_test):

    #on commence par initialiser les compteurs
    TP = np.sum((codants_predits == 1) & (codants_test == 1))  #codants corrects
    FP = np.sum((codants_predits == 1) & (codants_test == 0))  #faux codants
    FN = np.sum((codants_predits == 0) & (codants_test == 1))  #faux non-codants
    TN = np.sum((codants_predits == 0) & (codants_test == 0))  #non-codants corrects

    #on construit ensuite la matrice de confusion
    matrice_confusion = np.array([[TP, FP], [FN, TN]])

    #etiquettes pour visualisation
    fig, ax = plt.subplots(figsize=(6, 6))
    cax = ax.matshow(matrice_confusion, cmap='Blues')
    fig.colorbar(cax)
    
    ax.set_xticklabels(['', 'non codant', 'codant'])
    ax.set_yticklabels(['', 'non codant', 'codant'])
    ax.set_xlabel('predictions')
    ax.set_ylabel('observations')
    
    #ajotu des annotations
    for (i, j), val in np.ndenumerate(matrice_confusion):
        ax.text(j, i, f'{val}', ha='center', va='center', color='red', fontsize=14)
    
    plt.title('matrice de confusion')
    plt.show()

    return matrice_confusion


""" Interprétations des résultats : L'accuracy - qui permet généralement de connaître les performances générales d'un modèle - est de 63,17% , ce qui indique que le modèle est en mesure de classer avec précision environ 63% des zones genomiques (codantes et non codantes) conformement aux commentaires. Ceci étant dis la précision du modèle semble légèrement supérieure à celle de la matrice de référence (exactitude de 63,17% contre 62,5%, à la nuance de couleur près), principalement grâce à une prédiction plus précise des zones codantes. Cependant, cela entraîne un cout supplementaire en temres de faux positifs. Cela indique que le modele pourrait anticiper davantage de genes codants, mais egalement de genes incorrects.

"""


def create_seq(N, Pi, A, B, states, obs):
    """
    cette fonction est faite pour générer une seq d'état cachés et une séquence d'obs.
    Parametres : N (=longueur de la seq a generer); Pi (=vecteur des proba init. des états) ;states(=liste des états possible); obs(=liste des observation)
    
    """
    #on initi les deux seq
    seq_etats = []
    seq_obs = []
    
    #on détermine l'état initial en fonction de Pi (probabilités initiales)
    etat_initial = np.random.choice(states, p=Pi)
    seq_etats.append(etat_initial)
    
    #on genzre les observations associes à cet état initial
    obs_initial = np.random.choice(obs, p=B[etat_initial])
    seq_obs.append(obs_initial)
    
    #on genere les etats/observations squi suivent
    for _ in range(1, N):
        # pour l'etat suivant base sur l'etat actuel + matrice de transition A
        etat_suivant = np.random.choice(states, p=A[seq_etats[-1]])
        seq_etats.append(etat_suivant)
        
        #observation associée à chaque état qui suit
        obs_suivant = np.random.choice(obs, p=B[etat_suivant])
        seq_obs.append(obs_suivant)
    
    return seq_etats, seq_obs


import numpy as np

def get_annoatation2(annotation_train):
    #cette fonction a pour but de transformer les annotations avec 4 états en annotations avec 10 états (etats étants transformés en prenant en compte les codons start et stop). en entree on aura le tableau d'annotations avec 4 états (=annotation_train) et en sortie le tableau d'annot à dix etats (annotatation_train2)
    
    annotation_train2 = []
    
    for anno in annotation_train:
        if anno == 0:
            #on ne change pas les etats intergenique
            annotation_train2.append(0)
        elif anno == 1:
            #cas des codon start , sont transformer en etats 1, 2, 3
            annotation_train2.extend([1, 2, 3])
        elif anno == 2:
            #les codons interne sont transf. en etats 4, 5, 6
            annotation_train2.extend([4, 5, 6])
        elif anno == 3:
            #les codons STOP en etats 7, 8, 9
            annotation_train2.extend([7, 8, 9])
    
    return np.array(annotation_train2)



