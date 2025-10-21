import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy import ndimage
import time as t


# --- Version OpenCV ---
def ouverture_opencv(image, structure):
    return cv2.morphologyEx(image.astype(np.uint8), cv2.MORPH_OPEN, structure, borderType=cv2.BORDER_CONSTANT, borderValue=0)

# --- Version SciPy ---
def ouverture_scipy(image, structure):
    eroded = ndimage.binary_erosion(image, structure=structure, border_value=0)
    opened = ndimage.binary_dilation(eroded, structure=structure, border_value=0)
    return opened.astype(np.uint8)

# Paramètres
structure = np.ones((3, 3), dtype=np.uint8)  # Structure impaire pour éviter les problèmes de centrage
time_end_cv = []
time_end_sp = []

for i in range(300):

    image = np.random.randint(0, 2, (256, 256)).astype(np.uint8)
    image_originale = image.copy()

    # OpenCV
    time_start = t.time()
    opened_cv = ouverture_opencv(image, structure)
    time_end_cv.append(t.time() - time_start)

    # SciPy
    time_start = t.time()
    opened_sp = ouverture_scipy(image, structure)
    time_end_sp.append(t.time() - time_start)



print(f"Temps moyen OpenCV  : {np.mean(time_end_cv):.6f} s ± {np.std(time_end_cv):.6f} s")
print(f"Temps moyen SciPy   : {np.mean(time_end_sp):.6f} s ± {np.std(time_end_sp):.6f} s")

# Calculer les différences
diff = np.abs(opened_sp.astype(int) - opened_cv.astype(int))
nb_diff = np.sum(diff)
print(f"Nombre de pixels différents : {nb_diff} / {image.size} ({100*nb_diff/image.size:.2f}%)")

# Affichage
fig, axs = plt.subplots(1, 4, figsize=(20, 5))
axs[0].imshow(image_originale, cmap='gray')
axs[0].set_title("Image avant filtre")
axs[0].axis('off')

axs[1].imshow(opened_cv, cmap='gray')
axs[1].set_title("Ouverture OpenCV")
axs[1].axis('off')

axs[2].imshow(opened_sp, cmap='gray')
axs[2].set_title("Ouverture SciPy")
axs[2].axis('off')

axs[3].imshow(diff, cmap='hot')
axs[3].set_title(f"Différence ({nb_diff} pixels)")
axs[3].axis('off')

plt.tight_layout()
plt.show()