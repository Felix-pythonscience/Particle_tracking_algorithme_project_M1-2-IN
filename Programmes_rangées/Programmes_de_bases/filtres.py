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