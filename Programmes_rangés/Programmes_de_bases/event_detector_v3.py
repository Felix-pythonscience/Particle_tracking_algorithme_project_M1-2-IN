# -*- coding: utf-8 -*-
"""
Created on Wed Oct 22 14:42:30 2025

@author: sebwi
"""

import numpy as np
from scipy.ndimage import label, sum as ndi_sum
#import matplotlib.pyplot as plt
from skimage.measure import regionprops

#%% Recupere le fichier

#%% Fonction de comptage alpha

def event_counting_alpha(alpha_matrix) :

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
    
    """
    # Matrice avec valeur du rapport pour chaque cluster
    overlap_matrix = labeled_matrix.copy()
    for i in np.unique(labeled_matrix):
        if i != 0:
            overlap_matrix[labeled_matrix == i] = estimated_counts[i - 1]
    """
    
    return aplha_count

#%% Fonction de comptage beta/muons/

def event_counting_electron_muon(electron_muon_matrix) :

    # Si matrice vide -> problème
    if not np.any(electron_muon_matrix):
        return 0
    
    # Critère discri muons 
    eccentricity_threshold=0.90
    solidity_threshold=0.95

    # Si matrice vide -> problème
    if np.sum(electron_muon_matrix) == 0:
        return 0
    
    # Labelise et compte le nombre de cluster trouvé
    structure = np.ones((3, 3), dtype=int)  # crée une matrice 2D 3c et 3l de 1 qui correspond aux 8 positions possibles autour du pixel observé
    labeled_matrix, num_clusters = label(electron_muon_matrix, structure=structure)     # fonction de scipy pour compter les cluster et avoir une matrice avec chaque cluster labelisé

    # Variables qui contiendront le nombre d'électrons et de muons 
    muon_count = 0
    electron_count = 0
    
    # Discrimination
    for props in regionprops(labeled_matrix, intensity_image=electron_muon_matrix):
        if props.label == 0:
            continue
        is_muon = (props.solidity >= solidity_threshold) and \
                  (props.eccentricity >= eccentricity_threshold)
        
        if is_muon:
            muon_count += 1
        else:
            electron_count += 1

    return electron_count, muon_count


#%% Fonction de comptage photons

def event_counting_photon(photon_matrix) :
    
    structure = np.ones((3, 3), dtype=int)
    labeled_matrix, photon_count = label(photon_matrix, structure=structure)
    
    return photon_count


#%% Graphique
"""
plt.figure(figsize=(8,8))
plt.imshow(df, cmap='grey', origin='upper')
plt.xlabel("x")
plt.ylabel("y")
plt.title("Alpha overlap")
plt.colorbar(label='Cluster size / Median of all clusters (rounded)')

print(event_counting_electron_muon(df))
"""