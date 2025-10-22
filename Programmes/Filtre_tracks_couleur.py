import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from acquisition import read, slice,slice_Tot
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
    structure_circulaire = cv.getStructuringElement(cv.MORPH_RECT, (4, 4))

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

# Affichage : encapsulé dans une fonction et colorbar limitée à la première ligne
def plot_results(image_originale, image_alpha, image_tracks, image_without_tracks, image_couleur, figsize=(16, 12)):
    vmin = float(np.nanmin(image_couleur))
    vmax = float(np.nanmax(image_couleur))
    fig = plt.figure(figsize=figsize)
    # augmenter hspace et wspace pour plus d'espaces entre les images
    gs = fig.add_gridspec(2, 5, width_ratios=[1, 1, 1, 1, 0.06], height_ratios=[1, 1], hspace=0.45, wspace=0.35)

    # Axes pour la première ligne (images colorées)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    ax4 = fig.add_subplot(gs[0, 3])

    # Axes pour la deuxième ligne (images sans couleur)
    ax1b = fig.add_subplot(gs[1, 0])
    ax2b = fig.add_subplot(gs[1, 1])
    ax3b = fig.add_subplot(gs[1, 2])
    ax4b = fig.add_subplot(gs[1, 3])

    # Axe pour la colorbar : seulement la première ligne (gs[0,4])
    cax = fig.add_subplot(gs[0, 4])

    # image avant filtre (avec couleurs)
    im1 = ax1.imshow(image_originale * image_couleur, cmap='viridis', vmin=vmin, vmax=vmax)
    ax1.set_title("image avant filtre")

    # image après filtre (alpha) - couleur
    im2 = ax2.imshow(image_alpha * image_couleur, cmap='viridis', vmin=vmin, vmax=vmax)
    ax2.set_title("image alpha")

    # image après filtre (tracks) - couleur
    im3 = ax3.imshow(image_tracks * image_couleur, cmap='viridis', vmin=vmin, vmax=vmax)
    ax3.set_title("image tracks")

    # image sans tracks - couleur
    im4 = ax4.imshow(image_without_tracks * image_couleur, cmap='viridis', vmin=vmin, vmax=vmax)
    ax4.set_title("image gammas")

    # Deuxième ligne : mêmes images mais en grayscale (pas de colorbar)
    im1b = ax1b.imshow(image_originale, cmap='gray')
    ax1b.set_title("image avant filtre ")

    im2b = ax2b.imshow(image_alpha, cmap='gray')
    ax2b.set_title("image alphas")

    im3b = ax3b.imshow(image_tracks, cmap='gray')
    ax3b.set_title("image tracks")

    im4b = ax4b.imshow(image_without_tracks, cmap='gray')
    ax4b.set_title("image gammas")

    # Colorbar attachée à l'axe cax (qui couvre uniquement la première ligne)
    cbar = fig.colorbar(im1, cax=cax, orientation='vertical')
    cax.yaxis.tick_right()
    cbar.set_label('TOT')

    plt.tight_layout()
    plt.show()




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

if __name__ == "__main__":
    # Lecture des données et création de l'image binaire
    file = "C:/Users/Graziani/Desktop/Projet CEA/Particle_tracking_algorithme_project_M1-2-IN/Programmes/5min_beta_SrY_1.5cm_ground_source/5min_beta_SrY_1.5cm_ground_source_r3.t3pa"
    file = "C:/Users/Graziani/Desktop/Projet CEA/Particle_tracking_algorithme_project_M1-2-IN/DATA-20251022T080148Z-1-001/DATA/alpha/60sec_alpha_39kbq_2.5cm_r0.t3pa"
    data = read(file)
    d_time = max(data.iloc[:, 1]) / 500  # Diviser le temps

    image = slice(data.to_numpy(), d_time, d_time)

    image_couleur = slice_Tot(data.to_numpy(), 0, d_time)

    image_originale = image.copy()  # Sauvegarder l'image originale

    image_without_alpha, image_alpha = filtre_alpha(image)# Appliquer le filtre pour enlever les tracks

    image_without_tracks, image_tracks = filtre_tracks(image_without_alpha)# Appliquer le filtre pour enlever les tracks

    # Création du dossier de sortie (à côté du script)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, 'output')
    os.makedirs(output_dir, exist_ok=True)
    # Sauvegarder les images (PNG pour visualisation et NPY pour données brutes)
    alpha_png = os.path.join(output_dir, 'image_alpha.png')
    alpha_npy = os.path.join(output_dir, 'image_alpha_3.npy')
    tracks_png = os.path.join(output_dir, 'image_tracks.png')
    tracks_npy = os.path.join(output_dir, 'image_tracks_2.npy')
    without_tracks_png = os.path.join(output_dir, 'image_without_tracks.png')
    without_tracks_npy = os.path.join(output_dir, 'image_without_tracks_2.npy')

    # Déterminer vmin/vmax communs pour toutes les images (utiliser les données non colorées pour les échelles)


    vmin = float(np.nanmin(image_couleur))
    vmax = float(np.nanmax(image_couleur))
    save_npy(image_originale, os.path.join(output_dir, 'image_originale.npy'))
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


    # Appel de la fonction d'affichage
    plot_results(image_originale, image_alpha, image_tracks, image_without_tracks, image_couleur)



