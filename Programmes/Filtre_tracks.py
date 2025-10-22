import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from acquisition import read, slice,slice_Tot
import os




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
file = "C:/Users/Félix/Desktop/Programmation/Projet_cea/Particle_tracking_algorithme_project_M1-2-IN/DATA-20251022T080148Z-1-001/DATA/Combined_Am_SrY/2.5cm/2.5cm_r0.t3pa"
data = read(file)
d_time = max(data.iloc[:, 1]) / 500  # Diviser le temps

image = slice(data.to_numpy(), 0, d_time)

image_couleur = slice_Tot(data.to_numpy(), 0, d_time)

image_originale = image.copy()  # Sauvegarder l'image originale

image_without_alpha, image_alpha = filtre_alpha(image)# Appliquer le filtre pour enlever les tracks

image_without_tracks, image_tracks = filtre_tracks(image_without_alpha)# Appliquer le filtre pour enlever les tracks

# Création du dossier de sortie (à côté du script)
script_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(script_dir, 'output')
os.makedirs(output_dir, exist_ok=True)

# Fonctions utilitaires de sauvegarde
def save_gray_png(arr, path, vmin=None, vmax=None):
    # Normaliser pour affichage 0-255 si nécessaire en utilisant vmin/vmax communs si fournis
    a = arr.astype('float64')
    if vmin is None:
        vmin = float(a.min())
    if vmax is None:
        vmax = float(a.max())
    if vmax == vmin:
        norm = a - vmin
    else:
        norm = (a - vmin) / (vmax - vmin) * 255.0
    norm = np.clip(norm, 0, 255)
    cv.imwrite(path, norm.astype(np.uint8))

def save_npy(arr, path):
    np.save(path, arr)

# Sauvegarder les images (PNG pour visualisation et NPY pour données brutes)
alpha_png = os.path.join(output_dir, 'image_alpha.png')
alpha_npy = os.path.join(output_dir, 'image_alpha.npy')
tracks_png = os.path.join(output_dir, 'image_tracks.png')
tracks_npy = os.path.join(output_dir, 'image_tracks.npy')
without_tracks_png = os.path.join(output_dir, 'image_without_tracks.png')
without_tracks_npy = os.path.join(output_dir, 'image_without_tracks.npy')

# Déterminer vmin/vmax communs pour toutes les images (utiliser les données non colorées pour les échelles)
all_vals = np.concatenate([
    image_originale.flatten(),
    image_alpha.flatten(),
    image_tracks.flatten(),
    image_without_tracks.flatten()
])
vmin = float(np.nanmin(all_vals))
vmax = float(np.nanmax(all_vals))

# Sauvegarder les PNG en utilisant la même échelle
save_gray_png(image_alpha * image_couleur, alpha_png, vmin=vmin, vmax=vmax)
save_npy(image_alpha, alpha_npy)

save_gray_png(image_tracks * image_couleur, tracks_png, vmin=vmin, vmax=vmax)
save_npy(image_tracks, tracks_npy)

save_gray_png(image_without_tracks * image_couleur, without_tracks_png, vmin=vmin, vmax=vmax)
save_npy(image_without_tracks, without_tracks_npy)

print('Saved:', alpha_png, alpha_npy)
print('Saved:', tracks_png, tracks_npy)
print('Saved:', without_tracks_png, without_tracks_npy)


# Créer les subplots
fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(12, 5))

# image avant filtre
im1 = ax1.imshow(image_originale * image_couleur, cmap='viridis', vmin=vmin, vmax=vmax)
ax1.set_title("image avant filtre")


# image après filtre (alpha)
im2 = ax2.imshow(image_alpha * image_couleur, cmap='viridis', vmin=vmin, vmax=vmax)
ax2.set_title("image après filtre (alpha)")


# image après filtre (tracks)
im3 = ax3.imshow(image_tracks * image_couleur, cmap='viridis', vmin=vmin, vmax=vmax)
ax3.set_title("image après filtre (tracks)")


# image sans tracks
im4 = ax4.imshow(image_without_tracks * image_couleur, cmap='viridis', vmin=vmin, vmax=vmax)
ax4.set_title("image après filtre (sans tracks)")

# Ajouter une colorbar partagée
cbar = fig.colorbar(im4, ax=[ax1, ax2, ax3, ax4], orientation='vertical', fraction=0.02, pad=0.04)
cbar.set_label('Counts')


plt.tight_layout()
plt.show()



