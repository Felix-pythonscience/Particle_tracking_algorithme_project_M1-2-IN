import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from acquisition import read, slice,slice_Tot
from scipy.ndimage import label
from Filtre_tracks_couleur import plot_results
import os


def ouverture(image, structure, anchor=None):
    # anchor : tuple (x,y) pour positionner le centre du kernel (permet de tester plusieurs centres)
    return cv.morphologyEx(image.astype(np.uint8), cv.MORPH_OPEN, structure, anchor=anchor, borderType=cv.BORDER_CONSTANT, borderValue=0)
# --- Version OpenCV 3 ---
def ouverture_erode(image, structure, anchor=None):
    opened = cv.morphologyEx(image.astype(np.uint8), cv.MORPH_OPEN, structure, anchor=anchor, borderType=cv.BORDER_CONSTANT, borderValue=0)
    eroded = cv.erode(image.astype(np.uint8), structure, anchor=anchor, borderType=cv.BORDER_CONSTANT, borderValue=0)

def filtre_alpha(image):
    # Kernel rectangulaire (taille 4x4 pour pouvoir tester 4 centres : (1,1),(1,2),(2,1),(2,2))
    structure_circulaire = cv.getStructuringElement(cv.MORPH_ELLIPSE, (4, 4))

    # Travailler sur une image binaire dérivée de l'image d'entrée
    binary = (image > 0).astype(np.uint8)

    # Tester les 4 ancres centrales et fusionner les ouvertures
    anchors = [(1, 1), (1, 2), (2, 1), (2, 2)]
    opened_list = [ouverture(binary, structure=structure_circulaire, anchor=anc) for anc in anchors]
    opened = np.maximum.reduce(opened_list).astype(np.uint8)

    # Appliquer le masque sur l'image originale (si image contient des comptes, on conserve les valeurs)
    image_alpha = image * opened
    image_without_alpha = image - image_alpha
    return image_without_alpha, image_alpha

def filtre_tracks(image):

    # Kernel vertical et horizontal
    structure_verticale = np.ones((1, 3), dtype=np.uint8)
    structure_horizontale = np.ones((3, 1), dtype=np.uint8)

    # Travailler sur une image binaire dérivée de l'image d'entrée
    binary = (image > 0).astype(np.uint8)

    # Ouverture
    opened_verticale = ouverture(binary, structure=structure_verticale)
    opened_horizontale = ouverture(binary, structure=structure_horizontale)

    # Dilater les ouvertures pour retrouver les tracks selon l'orientation opposée
    dil1 = cv.dilate(opened_verticale, structure_horizontale, iterations=1)
    dil2 = cv.dilate(opened_horizontale, structure_verticale, iterations=1)

    mask = np.maximum(dil1, dil2).astype(np.uint8)

    # Appliquer le masque sur l'image originale (si image contient des comptes, on conserve les valeurs)
    image_tracks = image * mask
    image_without_tracks = image - image_tracks
    return image_without_tracks, image_tracks
def compteur_particles(file = "None", t= 0, d_time = None,plot = False):
    data = file if not(type(file) == str) else read(file)
    d_time = d_time if not d_time==None else max(data.iloc[:, 1]) / 100  # Diviser le temps

    image = slice(data.to_numpy(), 0, d_time)

    

    

    image_without_alpha, image_alpha = filtre_alpha(image)# Appliquer le filtre pour enlever les tracks

    image_without_tracks, image_tracks = filtre_tracks(image_without_alpha)# Appliquer le filtre pour enlever les tracks

    N_alpha = label(image_alpha)[1]
    N_tracks = label(image_tracks)[1]   
    N_gamma = label(image_without_tracks)[1]
    if plot:
        image_originale = image # Sauvegarder l'image originale
        image_couleur = slice_Tot(data.to_numpy(), 0, d_time)
        plot_results(image_originale, image_alpha, image_tracks, image_without_tracks, image_couleur)
    return N_alpha, N_tracks, N_gamma
if __name__ == "__main__":
    # Lecture des données et création de l'image binaire
    file = "C:/Users/Graziani/Desktop/Projet CEA/Particle_tracking_algorithme_project_M1-2-IN/DATA-20251022T080148Z-1-001/DATA/alpha/60sec_alpha_39kbq_2.5cm_r0.t3pa"
    file = "C:/Users/Graziani/Desktop/Projet CEA/Particle_tracking_algorithme_project_M1-2-IN/DATA-20251022T080148Z-1-001/DATA/Combined_Am_SrY/2.5cm/2.5cm_r0.t3pa"


    N_alpha, N_tracks, N_gamma = compteur_particles(file =file, plot=True)

    print(f"Nombre de particules alpha détectées : {N_alpha}")
    print(f"Nombre de particules tracks détectées : {N_tracks}")    
    print(f"Nombre de particules gamma détectées : {N_gamma}")

