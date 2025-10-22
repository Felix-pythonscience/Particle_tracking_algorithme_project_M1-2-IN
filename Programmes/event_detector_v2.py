# -*- coding: utf-8 -*-
"""
Created on Wed Oct 22 14:42:30 2025

@author: crphyA1
"""
#%%
import numpy as np
from scipy.ndimage import label, sum as ndi_sum
import matplotlib.pyplot as plt

#%%
df = np.load("image_alpha_3.npy")
df_full = np.load("image_originale.npy")

#%%
def compter_evenements_chevauchants(matrice: np.ndarray) -> int:

    # Si matrice vide -> problème
    if np.sum(matrice) == 0:
        return 0

    # Labelise et compte le nombre de cluster trouvé sans critere de chevauchement 
    structure = np.ones((3, 3), dtype=int)  # crée une matrice 2D de 3 sur 3 remplie 1 qui correspond aux 8 positions possibles autour du pixel observé
    labeled_matrix, num_clusters = label(matrice, structure=structure)  

    # Calcul de la taille de chaque cluster.
    labels = np.arange(1, num_clusters + 1)     # liste avec les indices de chaque cluster (1,...,nmax de cluster)
    sizes = ndi_sum(matrice, labeled_matrix, labels)    # liste de la taille de chaque cluster
    #print(sizes)
    
    # Estimation taille mediane des alphas
    typical_size = np.median(sizes)     # on recupere la mediane des tailles de cluster pour pouvoir compter correctement les chevauchement 
    #print(typical_size)

    # Prise en compte des chevauchements
    estimated_counts = np.round(sizes / typical_size)   # arrondi de la taille des cluster par rapport a la taille mediane
    print(estimated_counts)
    total_events = int(np.sum(estimated_counts))    # valeur du comptage avec prise en compte du chevauchement
    
    return total_events, labeled_matrix




n_event,labeled_matrix = compter_evenements_chevauchants(df)
print(n_event)




plt.figure(figsize=(15,15))
plt.imshow(df_full, cmap='gray', origin='upper')  # origin='upper' pour que (0,0) soit en haut à gauche
plt.title("Matrice de tout")
plt.xlabel("y")
plt.ylabel("x")
plt.colorbar(label='Valeur')
plt.show()

plt.figure(figsize=(15,15))
plt.imshow(df, cmap='gray', origin='upper')  # origin='upper' pour que (0,0) soit en haut à gauche
plt.title("Matrice des alphas")
plt.xlabel("y")
plt.ylabel("x")
plt.colorbar(label='Valeur')
plt.show()