# -*- coding: utf-8 -*-
"""
Created on Wed Oct 22 14:42:30 2025

@author: sebwi
"""

import numpy as np
from scipy.ndimage import label, sum as ndi_sum
import matplotlib.pyplot as plt
from skimage.measure import regionprops


def event_counting_alpha(alpha_matrix, plot_result=True) :

    # Si matrice vide -> problème
    if np.sum(alpha_matrix) == 0:
        return 0

    # Labelise et compte le nombre de cluster trouvé sans critere de chevauchement 
    structure = np.ones((3, 3), dtype=int)  # crée une matrice 2D de 3 sur 3 remplie 1 qui correspond aux 8 positions possibles autour du pixel observé
    labeled_matrix, num_clusters = label(alpha_matrix, structure=structure)      # fonction de scipy pour compter les cluster et avoir une matrice avec chaque cluster labelisé
    #print("Nombre de cluster sans filtre de chevauchement : ", num_clusters, "\n")

    # Calcul de la taille de chaque cluster
    labels = np.arange(1, num_clusters + 1)     # liste avec les indices de chaque cluster (1,...,nmax de cluster)
    #print("Label de chaque cluster : \n", labels, "\n") 
    sizes = ndi_sum(alpha_matrix, labeled_matrix, labels)    # liste de la taille de chaque cluster
    #print("Taille des clusters : \n",sizes, "\n")
    
    # Estimation taille mediane des alphas
    typical_size = np.median(sizes)     # on recupere la mediane des tailles de cluster pour pouvoir compter correctement les chevauchement 
    #print("Taille médiane : ", typical_size, "\n")
    
    # Prise en compte des chevauchements
    estimated_counts = np.round(sizes / typical_size)   # liste de l'arrondi de la taille des cluster par rapport a la taille mediane
    estimated_counts[estimated_counts == 0] = 1     # transforme les arrondis 0 en 1 
    #print("Liste des rapport de taille avec la médiane : \n", estimated_counts, "\n")
    aplha_count = int(np.sum(estimated_counts))    # valeur du comptage avec prise en compte du chevauchement  (somme de la liste estimated_sounts)
    #print("Nombre de cluster avec filtre de chevauchement : ", aplha_count, "\n")
    
    
    # Affichage du graphique optionel des chevauchements
    if plot_result:   
        overlap_matrix = labeled_matrix.copy()
        for i in np.unique(labeled_matrix):
            if i != 0:
                overlap_matrix[labeled_matrix == i] = estimated_counts[i - 1]
        plt.figure(figsize=(10,10))
        plt.imshow(overlap_matrix, cmap='hot', origin='upper')
        plt.title("Classification des chevauchements")
        plt.xlabel("X")
        plt.ylabel("Y")
    
    return aplha_count

#%% Fonction de comptage beta/muons/

def event_counting_electron_muon(electron_muon_matrix, plot_result=True) :

    # Si matrice vide -> problème
    if not np.any(electron_muon_matrix):
        return 0
    
    # Critère discri 
    eccentricity_threshold = 0.99
    solidity_threshold = 0.99
    area_threshold = 10
    
    # Labelise et compte le nombre de cluster trouvé
    structure = np.ones((3, 3), dtype=int)  # crée une matrice 2D 3c et 3l de 1 qui correspond aux 8 positions possibles autour du pixel observé
    labeled_matrix, num_clusters = label(electron_muon_matrix, structure=structure)     # fonction de scipy pour compter les cluster et avoir une matrice avec chaque cluster labelisé

    # Variables qui contiendront le nombre d'électrons, de muons et d'alphas
    muon_count = 0
    electron_count = 0
    alpha_count = 0
    
    # Matrice pour visualiser la discrimination
    classification_matrix = np.zeros_like(labeled_matrix, dtype=int)
    
    # Discrimination
    for props in regionprops(labeled_matrix, intensity_image=electron_muon_matrix):     # on regarde chaque cluster
        is_muon = (props.eccentricity >= eccentricity_threshold) and (props.area > area_threshold)    # on vérifie les conditions solidi/eccentri (booléen)
        is_alpha = (props.solidity >= solidity_threshold) and (props.area > area_threshold)
        #print(is_muon, "Soli :", props.solidity, "   Eccentri :", props.eccentricity)
        
        if is_muon:
            muon_count += 1
            classification_matrix[labeled_matrix == props.label] = 2
            print("muon", props.eccentricity)
        elif is_alpha:
            alpha_count += 1
            classification_matrix[labeled_matrix == props.label] = 1
            print("alpha", props.solidity)
        else:
            electron_count += 1
            classification_matrix[labeled_matrix == props.label] = 3

    # Affichage du graphique optionel des discriminations
    if plot_result:
        plt.figure(figsize=(10,10))
        plt.imshow(classification_matrix, cmap='hot', origin='upper')
        plt.title(f"Classification des particules (Électrons: {electron_count}, Muons: {muon_count}, Alphas: {alpha_count})")
        plt.xlabel("X")
        plt.ylabel("Y")
        
    return electron_count, muon_count, alpha_count


#%% Fonction de comptage photons

def event_counting_photon(photon_matrix) :
    
    structure = np.ones((3, 3), dtype=int)
    labeled_matrix, photon_count = label(photon_matrix, structure=structure)
    
    return photon_count


