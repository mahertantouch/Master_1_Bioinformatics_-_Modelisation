#maher Tantouch groupe 4 bmc-bim
import numpy as np
import matplotlib.pyplot as plt

import numpy as np

def gen_data_lin(a, b, sig, N, Ntest):
    """
    #fonction a pour but de générer des données jouets qui se basent sur la relation linéaire entre la v.a indépendante X et une variable dépendante y (avec bruit gaussien ajouté).

    #entree : poyr rappel : a est la pente de la droite linéaire, b l'intersection avec cette droite, sig l'écart type du bruit, N le nombre de points pour l'ensemble d'apprentissage , et Ntest le nombre de points pour l'ensemble de test.


    #sortie : pour rappel :  X_train correspond aux abscisses de l'ensemble d'apprentissage ,  y_train les ordonnées correspondantes (avec bruit) ;  X_test : abscisses de l'ensemble de test trié et ; et y_test les ordonnées correspondantes avec le bruit.
    """
    #abcisses triees
    X_train = np.random.rand(N) * 10  # random.rand sort une valeur entre 0 et 1, en multipliant par 10 on obtient une valeur entre 0 et 10.
    X_train.sort()  #poir trier les points

    #ordonnées avec le modele lineaire et bruit
    y_train = a * X_train + b + np.random.randn(N) * sig

    #abscisses pour l'ensemble de test
    X_test = np.random.rand(Ntest) * 10
    X_test.sort()

    #ordonnees pour l'ensemble de test avec le meme modele et bruit
    y_test = a * X_test + b + np.random.randn(Ntest) * sig

    return X_train, y_train, X_test, y_test

def modele_lin_analytique(X, y):
   
    #on cimmence par faire la moyenne des donnees:
    mean_X = np.mean(X)
    mean_y = np.mean(y)

    #on determine la covariance empirique (non corrigee) entre X et y
    cov_Xy = np.cov(X, y, bias=True)[0, 1]

    #on calcule la variance empirique (non corrigee) de X
    var_X = np.var(X)

    #on calcul les parametres du modele lineaire
    ahat = cov_Xy / var_X  # Pente
    bhat = mean_y - ahat * mean_X  # intersection

    return ahat, bhat


def calcul_prediction_lin(X, a, b):
    #calcule des prédictions lineaires pour une variable X:
    
    yhat = a * X + b
    return yhat

def erreur_mc(y, yhat):
    #erreur au sens des moindres carrés entre les valeurs reelles et predites:
    erreur = np.mean((y - yhat) ** 2)
    return erreur


def dessine_reg_lin(X_train, y_train, X_test, y_test, a, b):
    #def de la droite de regression
    X_line = np.linspace(min(np.min(X_train), np.min(X_test)), 
                         max(np.max(X_train), np.max(X_test)), 100)
    y_line = a * X_line + b

    #nuages de points
    plt.figure(figsize=(8, 6))
    plt.plot(X_train, y_train, 'bo-', alpha=0.6, label='train', markersize=6, linewidth=1)
    plt.scatter(X_test, y_test, color='red', alpha=0.15, label='test')

    #drpite de regression
    plt.plot(X_line, y_line, color='green', label='predictizn', linewidth=2)

    #legende du graphique
        #limites des axes
    plt.xlim(0, 0.6)
    plt.ylim(-2, 6)
        #legendes
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title('Régression Linéaire : Données et Droite Ajustée')
    plt.legend()
    plt.grid(True)
    plt.show()

def make_mat_lin_biais(X_train):
    #on genere la matrice X enrichie en ajoutant une colonne de 1 pour le biais.
    Xe = np.vstack((X_train, np.ones(X_train.shape[0]))).T
    return Xe

def reglin_matriciel(Xe, y_train):
     #ii on calcule les paramètres w (biais et coefficient) de la régression linéaire en utilisant la méthode matricielle (moindres carrés).
    
    #A = X^T *X
    A = np.dot(Xe.T, Xe)
    
    #B = X^T *y
    B = np.dot(Xe.T, y_train)
    
    # rzsolution du systeme d'eq lin
    w = np.linalg.solve(A, B)
    
    return w

def calcul_prediction_matriciel(Xe, w):
    #but :determune les predictions pour les données Xe à partir des parametres w.
    yhat = np.dot(Xe, w)
    return yhat

def gen_data_poly2(a, b, c, sig, N, Ntest):
    #genere les donnees d'apprentissage et de test pour une regress polynomiale de degre 2 (en utilisant y = ax^2 + bx + c + eps)
    
    # Xp dans l'intervalle [0, 1]
    Xp_train = np.random.uniform(0, 1, N)
    Xp_test = np.random.uniform(0, 1, Ntest)
    
    #bruit aléatoire
    eps_train = np.random.normal(0, sig, Xp_train.shape)
    eps_test = np.random.normal(0, sig, Xp_test.shape)
    
    #calcyles des y = ax^2 + bx + c + bruit
    yp_train = a * Xp_train**2 + b * Xp_train + c + eps_train
    yp_test = a * Xp_test**2 + b * Xp_test + c + eps_test
    
    return Xp_train, yp_train, Xp_test, yp_test

    #legende graphe
    plt.plot(Xp_test, yp_test, 'r.', alpha=0.2, label="test")
    plt.plot(Xp_train, yp_train, 'b.', label="train")
    plt.legend()
    plt.show()

def make_mat_poly_biais(X):
    Xe = np.column_stack((X, X**2, np.ones(X.shape[0])))  #permet d'ajouter X, X^2 et une colonne de biais
    return Xe


def dessine_poly_matriciel(Xp_train, yp_train, Xp_test, yp_test, w):
    """
    trace les donnees polynomiales et la solution de regression polynomiale

    nouvel argument w : coefficients obtenus via la regression polynomiale

    affiche un graphe des donnees et de la solution
    """
    # predictions pour l'apprentissage et le test
    Xe_train = make_mat_poly_biais(Xp_train)
    Xe_test = make_mat_poly_biais(Xp_test)

    yhat_train = calcul_prediction_matriciel(Xe_train, w)
    yhat_test = calcul_prediction_matriciel(Xe_test, w)

    # calcul des erreurs au sens des moindres carres
    erreur_train = erreur_mc(yhat_train, yp_train)
    erreur_test = erreur_mc(yhat_test, yp_test)

    # affichage des erreurs
    print(f'erreur moyenne au sens des moindres carres (train): {erreur_train:.4}')
    print(f'erreur moyenne au sens des moindres carres (test): {erreur_test:.4}')

    # trace des donnees
    plt.figure(figsize=(8, 6))
    plt.scatter(Xp_train, yp_train, color='blue', alpha=0.6, label='donnees d\'apprentissage')
    plt.scatter(Xp_test, yp_test, color='red', alpha=0.3, label='donnees de test')

    # trace des solutions estimees
    X_plot = np.linspace(0, 1, 500).reshape(-1, 1)
    Xe_plot = make_mat_poly_biais(X_plot)
    yhat_plot = calcul_prediction_matriciel(Xe_plot, w)

    plt.plot(X_plot, yhat_plot, color='green', label='solution estimee')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('regression polynomiale')
    plt.legend()
    plt.grid(True)
    plt.show()






