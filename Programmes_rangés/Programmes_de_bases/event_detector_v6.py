# -*- coding: utf-8 -*-
"""
Created on Wed Oct 22 14:42:30 2025

@author: sebwi
"""

import numpy as np
from scipy.ndimage import label, sum as ndi_sum
import matplotlib.pyplot as plt
from skimage.measure import regionprops_table


def event_counting_alpha(alpha_matrix, 
                         plot_result=False) :

    # Si matrice vide -> problème / initialisation
    if not np.any(alpha_matrix):
        return 0    
    alpha_count = 0
    
    # Labelise et compte le nombre de cluster trouvé sans critere de chevauchement 
    structure = np.ones((3, 3), dtype=int)  # crée une matrice 2D de 3 sur 3 remplie 1 qui correspond aux 8 positions possibles autour du pixel observé
    labeled_matrix, num_clusters = label(alpha_matrix, structure=structure)      # fonction de scipy pour compter les cluster et avoir une matrice avec chaque cluster labelisé

    # Calcul de la taille de chaque cluster
    labels = np.arange(1, num_clusters + 1)     # liste avec les indices de chaque cluster (1,...,nmax de cluster)
    sizes = ndi_sum(alpha_matrix, labeled_matrix, labels)    # liste de la taille de chaque cluster
    
    # Masque pour filtrer les clusters trop petits qui ne sont pas supposés être la
    valid_clusters_mask = (sizes >= 2)   # masque pour taille minimale des cluster subjective, ici 2
    valid_sizes = sizes[valid_clusters_mask]    # liste des tailles corrigée
    
    # Estimation taille mediane des alphas
    typical_size = np.median(valid_sizes)     # on recupere la mediane des tailles de cluster pour pouvoir compter correctement les chevauchement 
    
    # Prise en compte des chevauchements
    estimated_counts = np.round(valid_sizes / typical_size)   # liste de l'arrondi de la taille des cluster par rapport a la taille mediane
    estimated_counts[estimated_counts == 0] = 1     # transforme les arrondis 0 en 1 
    alpha_count = int(np.sum(estimated_counts))    # valeur du comptage avec prise en compte du chevauchement  (somme de la liste estimated_sounts)
    
    # Affichage du graphique optionel des chevauchements
    if plot_result:   
        lookup_table = np.concatenate(([0], estimated_counts))
        overlap_matrix = lookup_table[labeled_matrix]
        plt.figure(figsize=(10, 10))
        plt.imshow(overlap_matrix, cmap='turbo', origin='upper')
        plt.title("Classification des chevauchements")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.colorbar(label="Nombre d'alphas estimés par cluster")
        plt.show()
    
    return alpha_count

#%% Fonction de comptage beta/muons/

def event_counting_electron_muon(electron_muon_matrix, plot_result=False,
                                eccentricity_threshold_muon = 0.99,
                                area_threshold_muon = 25,
                                eccentricity_threshold_alpha = 0.70,
                                solidity_threshold_alpha = 1,
                                area_threshold_alpha = 9) :

    # Si matrice vide -> problème
    if not np.any(electron_muon_matrix):
        return 0,0,0

    # Labelise et compte le nombre de cluster trouvé
    structure = np.ones((3, 3), dtype=int)  # crée une matrice 2D 3c et 3l de 1 qui correspond aux 8 positions possibles autour du pixel observé
    labeled_matrix, num_clusters = label(electron_muon_matrix, structure=structure)     # fonction de scipy pour compter les cluster et avoir une matrice avec chaque cluster labelisé


    properties_needed = ['label', 'eccentricity', 'solidity', 'area']
    props_table = regionprops_table(labeled_matrix, intensity_image=electron_muon_matrix, properties=properties_needed)
    
    labels = props_table['label']
    eccentricities = props_table['eccentricity']
    solidities = props_table['solidity']
    areas = props_table['area']
    
    is_muon_vec = (eccentricities >= eccentricity_threshold_muon) & (areas > area_threshold_muon)
    is_alpha_vec = (solidities >= solidity_threshold_alpha) & (eccentricities <= eccentricity_threshold_alpha) & (areas > area_threshold_alpha)
    
    class_per_label = np.full(labels.shape, 1, dtype=int)
    class_per_label[is_muon_vec] = 2    
    class_per_label[is_alpha_vec] = 3

    electron_count = 0
    muon_count = 0
    alpha_count = 0

    electron_count = int(np.sum(class_per_label == 1))
    muon_count = int(np.sum(class_per_label == 2))
    alpha_count = int(np.sum(class_per_label == 3))

    mapping = np.zeros(num_clusters + 1, dtype=int)
    mapping[labels] = class_per_label
    classification_matrix = mapping[labeled_matrix]
    
    alpha_positions = np.argwhere(classification_matrix == 3)

    if plot_result:
        plt.figure(figsize=(10,10))
        plt.imshow(classification_matrix, cmap='turbo', origin='upper')
        plt.title(f"Classification des particules (Électrons: {electron_count}, Muons: {muon_count}, Alphas: {alpha_count})")
        plt.xlabel("X")
        plt.ylabel("Y")
    
    return electron_count, muon_count, alpha_count, alpha_positions


#%% Fonction de comptage photons

def event_counting_photon(photon_matrix) :
    
    # Si matrice vide -> problème
    if not np.any(photon_matrix):
        return 0    
    
    structure = np.ones((3, 3), dtype=int)
    photon_count = label(photon_matrix, structure=structure)[1]
    
    return photon_count

