import numpy as np
import cv2 as cv



#Fonctions d'ouverures morphologiques
def ouverture(image, structure, anchor=None):
    """Perform a morphological opening on an image using the given structuring element.

    Parameters
    ----------
    image : ndarray
        Input image (will be cast to uint8).
    structure : ndarray
        Structuring element (kernel) used for the morphological operation.
    anchor : tuple, optional
        Anchor point (x, y) inside the kernel to shift the kernel center.

    Returns
    -------
    ndarray
        The opened image (uint8).
    """
    # anchor : tuple (x,y) pour positionner le centre du kernel (permet de tester plusieurs centres)
    return cv.morphologyEx(image.astype(np.uint8), cv.MORPH_OPEN, structure, anchor=anchor, borderType=cv.BORDER_REFLECT)

def ouverture_erode(image, structure, anchor=None):
    """Apply opening then erosion to the image and return both results.

    This helper returns both the opened and eroded images. The function is
    primarily used for experimenting with morphological pipelines.

    Parameters
    ----------
    image : ndarray
        Input image to process.
    structure : ndarray
        Structuring element for morphology.
    anchor : tuple, optional
        Anchor point to shift the structuring element.

    Returns
    -------
    tuple
        (opened, eroded) images as uint8 arrays.
    """
    opened = cv.morphologyEx(image.astype(np.uint8), cv.MORPH_OPEN, structure, anchor=anchor, borderType=cv.BORDER_REFLECT)
    eroded = cv.erode(image.astype(np.uint8), structure, anchor=anchor, borderType=cv.BORDER_REFLECT)
    return opened,eroded # si besoin de retourner l'image érodée

def filtre_alpha(image):
    # Kernel rectangulaire (taille 4x4 pour pouvoir tester 4 centres : (1,1),(1,2),(2,1),(2,2))
    structure_circulaire = cv.getStructuringElement(cv.MORPH_RECT, (4, 4))

    # Travailler sur une image binaire dérivée de l'image d'entrée
    binary = (image > 0).astype(np.uint8)

    # Tester les 4 ancres centrales et fusionner les ouvertures
    anchors = [(1, 1), (1, 2), (2, 1), (2, 2)]
    opened_list = [ouverture(binary, structure=structure_circulaire, anchor=anc) for anc in anchors]
    opened = np.maximum.reduce(opened_list).astype(np.uint8)

    # Dilater légèrement le masque alpha pour regrouper/élargir les petites régions
    # Utilise un élément structurant 3x3 (rectangle) pour une dilatation isotrope
    kernel_3x3 = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    opened = cv.dilate(opened, kernel_3x3, iterations=1)

    # Appliquer le masque sur l'image originale (si image contient des comptes, on conserve les valeurs)
    image_alpha = image * opened
    image_without_alpha = image - image_alpha
    return image_without_alpha, image_alpha

def filtre_tracks(image, falses_alphas:np.ndarray = None):
    """
    Morphological filtering to separate tracks (electrons/muons) from the image.
    Parameters
    ----------
    image : ndarray
        Input image to filter.
    falses_alphas : ndarray, optional
        Optional mask of false alphas to include in tracks.
    Returns
    -------
    tuple
        (image_without_tracks, image_tracks) as uint8 arrays.

    """
    ## Reconstruction of the false alpha matrix 
    ## Reconstruction of the false alpha matrix 
    if falses_alphas is not None:
        falses_alphas_image = np.zeros_like(image, dtype=np.uint8)
        # Version vectorisée : empile les listes de pixels puis indexe en masse
        try:
            # gestion si falses_alphas est vide
            if len(falses_alphas) == 0:
                stacked = np.empty((0, 2), dtype=int)
            else:
                stacked = np.vstack([np.asarray(fa) for fa in falses_alphas])
            if stacked.size:
                rows = stacked[:, 0].astype(np.intp)
                cols = stacked[:, 1].astype(np.intp)
                # sécurité bornes indices
                rows = np.clip(rows, 0, image.shape[0] - 1)
                cols = np.clip(cols, 0, image.shape[1] - 1)
                falses_alphas_image[rows, cols] = 1
        except Exception:
            # Fallback robuste si format inattendu
            for false_alpha in falses_alphas:
                for pixels in false_alpha:
                    falses_alphas_image[int(pixels[0]), int(pixels[1])] = 1
        image = np.maximum(image, falses_alphas_image)

    ## filtering of tracks
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
    kernel_3x3 = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    mask = cv.dilate(mask, kernel_3x3, iterations=1)
    # Appliquer le masque sur l'image originale (si image contient des comptes, on conserve les valeurs)
    image_tracks = image * mask
    image_without_tracks = image - image_tracks
    return image_without_tracks, image_tracks