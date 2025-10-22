import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from acquisition import read, slice




def ouverture(image, structure):
    return cv.morphologyEx(image.astype(np.uint8), cv.MORPH_OPEN, structure, borderType=cv.BORDER_CONSTANT, borderValue=0)
# --- Version OpenCV 3 ---
def ouverture_erode(image, structure):
    opened = cv.morphologyEx(image.astype(np.uint8), cv.MORPH_OPEN, structure, borderType=cv.BORDER_CONSTANT, borderValue=0)
    eroded = cv.erode(image.astype(np.uint8), structure, borderType=cv.BORDER_CONSTANT, borderValue=0)

def filtre_alpha(image):
    # Kernel circulaire
    structure_circulaire = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))

    # Travailler sur une image binaire dérivée de l'image d'entrée
    binary = (image > 0).astype(np.uint8)

    # Ouverture
    opened = ouverture(binary, structure=structure_circulaire)

    # Appliquer le masque sur l'image originale (si image contient des comptes, on conserve les valeurs)
    image_alpha = image * opened
    image_without_alpha = image - image_alpha
    return image_without_alpha, image_alpha

def filtre_tracks(image):
    # Kernel vertical et horizontal
    structure_verticale = np.ones((1, 4), dtype=np.uint8)
    structure_horizontale = np.ones((4, 1), dtype=np.uint8)

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


# Lecture des données et création de l'image binaire
file = "C:/Users/Félix/Desktop/Programmation/Projet_cea/Particle_tracking_algorithme_project_M1-2-IN/DATA-20251022T080148Z-1-001/DATA/Combined_Am_SrY/2.5cm/2.5cm_r3.t3pa"
data = read(file)
d_time = max(data.iloc[:, 1]) / 500  # Diviser le temps

image = slice(data.to_numpy(), 0, d_time)


image_originale = image.copy()  # Sauvegarder l'image originale

image_without_alpha, image_alpha = filtre_alpha(image)# Appliquer le filtre pour enlever les tracks

image_without_tracks, image_tracks = filtre_tracks(image_without_alpha)# Appliquer le filtre pour enlever les tracks


# Créer les subplots
fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(12, 5))

# image avant filtre
ax1.imshow(image_originale, cmap='gray')
ax1.set_title("image avant filtre")


# image après filtre (alpha)
ax2.imshow(image_alpha, cmap='gray')
ax2.set_title("image après filtre (alpha)")


# image après filtre (tracks)
ax3.imshow(image_tracks, cmap='gray')
ax3.set_title("image après filtre (tracks)")


# image sans tracks
ax4.imshow(image_without_tracks, cmap='gray')
ax4.set_title("image après filtre (sans tracks)")


plt.tight_layout()
plt.show()



