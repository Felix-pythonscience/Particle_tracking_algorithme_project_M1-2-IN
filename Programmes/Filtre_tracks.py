import numpy as np 
import matplotlib.pyplot as plt
from scipy import ndimage
import time as t
Time_end = []

def ouverture(image, structure):
    eroded = ndimage.binary_erosion(image, structure=structure)
    opened = ndimage.binary_dilation(eroded, structure=structure)
    return opened


Image = np.random.randint(0, 2, (256, 256))
Image_originale = Image.copy()  # Sauvegarder l'image originale

# Définir un élément structurant (ici, un carré 3x3)
structure_verticale = np.ones((1, 4), dtype=np.uint8)
structure_horizontale = np.ones((4, 1), dtype=np.uint8)

# Ouverture
opened_verticale = ouverture(Image, structure=structure_verticale)
opened_horizontale = ouverture(Image, structure=structure_horizontale)

# Combiner les deux ouvertures (union)
Image_verticale = Image * np.maximum(
    ndimage.binary_dilation(opened_verticale, structure=structure_horizontale),
    ndimage.binary_dilation(opened_horizontale, structure=structure_verticale))

# Créer les subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Image avant filtre
ax1.imshow(Image_originale, cmap='gray')
ax1.set_title("Image avant filtre")
ax1.axis('off')

# Image après filtre
ax2.imshow(Image_verticale, cmap='gray')
ax2.set_title("Image après filtre (tracks)")
ax2.axis('off')

plt.tight_layout()
plt.show()



