# -*- coding: utf-8 -*-
"""
Created on Fri Oct 24 14:09:20 2025

@author: crphyA1
"""
import numpy as np
from scipy.ndimage import label, sum as ndi_sum
from skimage.measure import regionprops
import matplotlib.pyplot as plt

#%%
df = np.load("muon_electron.npy")


#%%
def event_counting_electron_muon(electron_muon_matrix) :
    
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


#%%
print(event_counting_electron_muon(df))

plt.figure(figsize=(15,15))
plt.imshow(df, cmap='gray', origin='upper')  # origin='upper' pour que (0,0) soit en haut à gauche
plt.title("Matrice image")
plt.xlabel("y")
plt.ylabel("x")
plt.colorbar(label='Valeur')
plt.show()