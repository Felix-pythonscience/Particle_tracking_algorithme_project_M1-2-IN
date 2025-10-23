# -*- coding: utf-8 -*-
"""
Created on Wed Oct 22 14:42:30 2025

@author: sebwi
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import label, sum as ndi_sum

#%% Fonction de comptage alpha

def event_counting_alpha(alpha_matrix) :

    # Si matrice vide -> problème
    if np.sum(alpha_matrix) == 0:
        return 0

    # Labelise et compte le nombre de cluster trouvé sans critere de chevauchement 
    structure = np.ones((3, 3), dtype=int)  # crée une matrice 2D de 3 sur 3 remplie 1 qui correspond aux 8 positions possibles autour du pixel observé
    labeled_matrix, num_clusters = label(alpha_matrix, structure=structure)      # fonction de scipy pour compter les cluster et avoir une matrice avec chaque cluster labelisé
    #print("Nombre de cluster sans filtre de chevauchement : ", num_clusters, "\n")

    # Calcul de la taille de chaque cluster.
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
    total_events = int(np.sum(estimated_counts))    # valeur du comptage avec prise en compte du chevauchement  (somme de la liste estimated_sounts)
    #print("Nombre de cluster avec filtre de chevauchement : ", total_events, "\n")
    
    # Matrice avec valeur du rapport pour chaque cluster
    overlap_matrix = labeled_matrix.copy()
    for i in np.unique(labeled_matrix):
        if i != 0:
            overlap_matrix[labeled_matrix == i] = estimated_counts[i - 1]

    return total_events,overlap_matrix


if __name__ == "__main__":
    
#%% Recupere le fichier
    df = np.load("C:/Users/Félix/Desktop/Programmation/Projet_cea/Particle_tracking_algorithme_project_M1-2-IN/Programmes_rangés/Programmes_de_benchmark/Benchmark_Results/compteur2/Evolution_détections_en_fonction_de_dt/dt = 1 divisé par 1636/image_alpha.npy")

    total_events, overlap_matrix = event_counting_alpha(df)
    print("Comptage : ", total_events)
    # Optionnel : afficher la matrice de chevauchement
    plt.imshow(overlap_matrix, cmap='viridis', interpolation='nearest')
    cbar = plt.colorbar()
    cbar.ax.yaxis.set_label_text('Valeur', fontsize=14)
    plt.title("Matrice de chevauchement", fontsize=16)
    plt.show()
