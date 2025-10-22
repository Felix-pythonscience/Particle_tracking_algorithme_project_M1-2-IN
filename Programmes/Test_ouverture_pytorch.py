import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy import ndimage
import time as t


# --- Version OpenCV ---
def ouverture_opencv(image, structure):
    return cv2.morphologyEx(image.astype(np.uint8), cv2.MORPH_OPEN, structure, borderType=cv2.BORDER_CONSTANT, borderValue=0)
# --- Version OpenCV 3 ---
def ouverture_opencv_3(image, structure):
    opened = cv2.morphologyEx(image.astype(np.uint8), cv2.MORPH_OPEN, structure, borderType=cv2.BORDER_CONSTANT, borderValue=0)
    eroded = cv2.erode(image.astype(np.uint8), structure, borderType=cv2.BORDER_CONSTANT, borderValue=0)
    return opened,eroded
def ouverture_opencv_2(image, structure):
    erode = cv2.erode(cv2.dilate(image.astype(np.uint8), structure, borderType=cv2.BORDER_CONSTANT, borderValue=0), structure, borderType=cv2.BORDER_CONSTANT, borderValue=0)
    dilate = cv2.dilate(cv2.erode(image.astype(np.uint8), structure, borderType=cv2.BORDER_CONSTANT, borderValue=0), structure, borderType=cv2.BORDER_CONSTANT, borderValue=0)
    return dilate,erode
# --- Version SciPy ---
def ouverture_scipy(image, structure):
    eroded = ndimage.binary_erosion(image, structure=structure, border_value=0)
    opened = ndimage.binary_dilation(eroded, structure=structure, border_value=0)
    return opened.astype(np.uint8)

# Paramètres
structure = np.ones((3, 3), dtype=np.uint8)  # Structure impaire pour éviter les problèmes de centrage
time_end_cv = []
time_end_cv_2 = []
time_end_cv_3 = []

for i in range(3000):

    image = np.random.randint(0, 2, (256, 256)).astype(np.uint8)
    image_originale = image.copy()

    # OpenCV
    time_start = t.time()
    opened_cv = ouverture_opencv(image, structure)
    time_end_cv.append(t.time() - time_start)

    #openCV 2
    time_start = t.time()
    opened_cv_2, eroded_cv_2 = ouverture_opencv_2(image, structure)
    time_end_cv_2.append(t.time() - time_start)

    #openCV 3
    time_start = t.time()
    opened_cv_3, eroded_cv_3 = ouverture_opencv_3(image, structure)
    time_end_cv_3.append(t.time() - time_start)




print(f"Temps moyen OpenCV  : {np.mean(time_end_cv):.6f} s ± {np.std(time_end_cv):.6f} s")
print(f"Temps moyen OpenCV 2: {np.mean(time_end_cv_2):.6f} s ± {np.std(time_end_cv_2):.6f} s")
print(f"Temps moyen OpenCV 3: {np.mean(time_end_cv_3):.6f} s ± {np.std(time_end_cv_3):.6f} s")


# Affichage
fig, axs = plt.subplots(1, 3, figsize=(20, 5))
axs[0].imshow(image_originale, cmap='gray')
axs[0].set_title("Image avant filtre")
axs[0].axis('off')

axs[1].imshow(opened_cv, cmap='gray')
axs[1].set_title("Ouverture OpenCV")
axs[1].axis('off')

axs[2].imshow(opened_cv_2, cmap='gray')
axs[2].set_title("Ouverture OpenCV 2")
axs[2].axis('off')

plt.tight_layout()
plt.show()