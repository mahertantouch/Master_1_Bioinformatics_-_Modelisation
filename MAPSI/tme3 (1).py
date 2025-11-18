import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl


def learnML_parameters(X_train, Y_train):
    # on initialise des tableaux pour mu et sigma
    mu = np.zeros((10, 256)) 
    sig = np.zeros((10, 256))
    
    # pour le parcours des 9 classes
    for class_label in range(10):
        # on filtre ici les images de la classe actuelle
        class_images = X_train[Y_train == class_label]
        
        # on calcule la moyenne et l'écart-type pour chaque pixel
        mu[class_label] = np.mean(class_images, axis=0)
        sig[class_label] = np.var(class_images, axis=0)
    
    return mu, sig

def log_likelihood(image, mu, sigma2, defeps):
    # ici on traite des valeurs de sigma2 pour éviter la division par 0
    if defeps > 0:
        sigma2 = np.maximum(sigma2, defeps)
    elif defeps == -1:
        # pour remplacer par 1 pour la vraisemblance dans ce cas
        sigma2 = np.where(sigma2 == 0, 1, sigma2)
    
    # determination de la log-vraisemblance
    log_likelihood_value = -0.5 * np.sum(np.log(2 * np.pi * sigma2) + ((image - mu) ** 2) / sigma2)
    
    return log_likelihood_value

def classify_image(image, mu, sig, defeps):
    # poir le calcul la log-vraisemblance pour chaque classe
    log_likelihoods = [log_likelihood(image, mu[i], sig[i], defeps) for i in range(len(mu))]
    
    # détermination de l'indice de la classe avec la log-vraisemblance maximale
    predicted_class = np.argmax(log_likelihoods)
    
    return predicted_class

def classify_all_images(X, mu, sig, defeps):
    # pour déterminer la classe pour chaque image dans le tableau X
    Y_hat = np.array([classify_image(image, mu, sig, defeps) for image in X])
    return Y_hat

def matrice_confusion(Y, Y_hat):
    # on determine le nombre de classes
    C = len(np.unique(Y))
    
    # pour initialiser la matrice de confusion
    confusion_matrice = np.zeros((C, C), dtype=int)
    
    # enfin on remplit la matrice de confusion
    for true_label, predicted_label in zip(Y, Y_hat):
        confusion_matrice[true_label, predicted_label] += 1
    
    return confusion_matrice



def ClassificationRate(Y, Y_hat):
    # Calculer le nombre total de prédictions
    total_predictions = len(Y)
    
    # Compter le nombre de bonnes prédictions
    correct_predictions = np.sum(Y == Y_hat)
    
    # Calculer le taux de bonne classification
    classification_rate = correct_predictions / total_predictions
    
    return classification_rate


def classifTest(X_test, Y_test, mu, sig, defeps):
    # on commence par les classes pour les données de test
    Y_hat = classify_all_images(X_test, mu, sig, defeps)
    
    # on calcule ensuite le taux de bonne classification
    classification_rate = ClassificationRate(Y_test, Y_hat)
    
    # génération de la matrice de confusion
    confusion_matrix = matrice_confusion(Y_test, Y_hat)
    
    # identification les images mal classées
    mal_classees_indices = np.where(Y_test != Y_hat)[0]
    
    # pour afficher le taux de bonne classification et la matrice de confusion
    print(f"Taux de bonne classification: {classification_rate:.2f}")
    print("Matrice de confusion:")
    print(confusion_matrix)
    
    # on retourne les indices des images mal classées
    return mal_classees









