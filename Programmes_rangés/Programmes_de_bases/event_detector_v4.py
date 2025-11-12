# -*- coding: utf-8 -*-
"""
Created on Wed Oct 22 14:42:30 2025

@author: sebwi
"""

import numpy as np
from scipy.ndimage import label, sum as ndi_sum
import matplotlib.pyplot as plt
from skimage.measure import regionprops


def event_counting_alpha(uncorrected_alpha_matrix,
                         solidity_threshold=0.99,
                         plot_result=False) :

    # Si matrice vide -> problème
    if not np.any(uncorrected_alpha_matrix):
        return 0,0

    # Labelise et compte le nombre de cluster trouvé, sans discrimination ni de critere de chevauchement 
    structure = np.ones((3, 3), dtype=int)  # crée une matrice 2D de 3 sur 3 remplie 1 qui correspond aux 8 positions possibles autour du pixel observé
    labeled_uncorrected_alpha_matrix, num_clusters_uncorrected = label(uncorrected_alpha_matrix, structure=structure)      # fonction de scipy pour compter les cluster et avoir une matrice avec chaque cluster labelisé
    #print("Nombre de cluster sans filtre de chevauchement : ", num_clusters, "\n")

    # Matrice pour visualiser la discrimination alpha/electron
    classification_matrix = np.zeros_like(labeled_uncorrected_alpha_matrix, dtype=int)

    # Tri entre les vrais alphas et les électrons 
    for props in regionprops(labeled_uncorrected_alpha_matrix, intensity_image=uncorrected_alpha_matrix):
        is_alpha = (props.solidity >= solidity_threshold)
        if is_alpha:
            classification_matrix[labeled_uncorrected_alpha_matrix == props.label] = 1
        else:
            classification_matrix[labeled_uncorrected_alpha_matrix == props.label] = 2


    ### PARTIE COMPTAGE D'ALPHAS ET CHEVAUCHEMENT
    # Matrice avec uniquement les alphas
    alpha_only_matrix = uncorrected_alpha_matrix * classification_matrix==1
    
    # Si matrice vide 
    if not np.any(alpha_only_matrix):
        
        # Matrice avec clusters labélisés des alphas et nombre de clusters, sans correction des chevauchements 
        labeled_alpha_only_matrix, num_clusters_alpha = label(alpha_only_matrix, structure=structure)      # fonction de scipy pour compter les cluster et avoir une matrice avec chaque cluster labelisé
        
        # Calcul de la taille de chaque cluster
        labels_alpha = np.arange(1, num_clusters_alpha + 1)     # liste avec les indices de chaque cluster (1,...,nmax de cluster)
        #print("Label de chaque cluster : \n", labels, "\n")
        sizes = ndi_sum(alpha_only_matrix, labeled_alpha_only_matrix, labels_alpha)    # liste de la taille de chaque cluster
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
        overlap_matrix = labeled_alpha_only_matrix.copy()
        for i in np.unique(labeled_alpha_only_matrix):
            if i != 0:
                overlap_matrix[labeled_alpha_only_matrix == i] = estimated_counts[i - 1]
        plt.figure(figsize=(10,10))
        plt.imshow(overlap_matrix, cmap='hot', origin='upper')
        plt.title("Classification des chevauchements")
        plt.xlabel("X")
        plt.ylabel("Y")
    
    
    ## PARTIE EXTRACTION DES COORDONEES DES CLUSTERS D'ELECTRONS
    # Matrice avec uniquement les electrons
    electron_only_matrix = uncorrected_alpha_matrix * classification_matrix==2    
    
    # Matrice avec clusters labélisés des electrons et nombre de clusters
    labeled_electron_only_matrix, num_clusters_electron = label(electron_only_matrix, structure=structure)      # fonction de scipy pour compter les cluster et avoir une matrice avec chaque cluster labelisé
    
    # Coordeonnées de chaque pixel de chaque cluster 
    all_cluster_coords = [prop.coords for prop in regionprops(labeled_electron_only_matrix)]
    
    # Affichage du graphique optionel des electrons discriminés
    if plot_result:
        plt.figure(figsize=(10,10))
        plt.imshow(electron_only_matrix, cmap='hot', origin='upper')
        plt.title("Electrons discriminés")
        plt.xlabel("X")
        plt.ylabel("Y")
    
    return aplha_count, all_cluster_coords

#%% Fonction de comptage beta/muons/

def event_counting_electron_muon(electron_muon_matrix,
                                 eccentricity_threshold=0.99,
                                 solidity_threshold=0.99,
                                 area_threshold=10,
                                 plot_result=False) :

    # Si matrice vide -> problème
    if not np.any(electron_muon_matrix):
        return 0,0,0
    
    # Labelise et compte le nombre de cluster trouvé
    structure = np.ones((3, 3), dtype=int)  # crée une matrice 2D 3c et 3l de 1 qui correspond aux 8 positions possibles autour du pixel observé
    labeled_matrix, num_clusters = label(electron_muon_matrix, structure=structure)     # fonction de scipy pour compter les cluster et avoir une matrice avec chaque cluster labelisé

    # Compteurs muons/alphas/electons
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
            #print("muon", props.eccentricity)
        elif is_alpha:
            alpha_count += 1
            classification_matrix[labeled_matrix == props.label] = 1
            #print("alpha", props.solidity)
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
    
    # Si matrice vide -> problème
    if not np.any(photon_matrix):
        return 0    
    
    structure = np.ones((3, 3), dtype=int)
    labeled_matrix, photon_count = label(photon_matrix, structure=structure)
    
    return photon_count
