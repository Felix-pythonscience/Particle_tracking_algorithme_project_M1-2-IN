import numpy as np                # Importation du module numpy pour la manipulation de tableaux
import matplotlib.pyplot as plt   # Importation du module matplotlib pour l'affichage des images
from scipy import ndimage         # Importation du module scipy.ndimage pour les opérations morphologiques (utilisé ici pour comparaison)
import time as t                  # Importation du module time pour mesurer le temps d'exécution

def erosion(image, structure):
    h, w = image.shape           # Récupère la hauteur et la largeur de l'image
    sh, sw = structure.shape     # Récupère la hauteur et la largeur de l'élément structurant
    pad_h, pad_w = sh // 2, sw // 2   # Calcule le nombre de pixels à ajouter pour le padding
    # Ajoute un padding de zéros autour de l'image pour gérer les bords
    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)
    eroded = np.ones_like(image)     # Initialise l'image érodée avec des zéros
    for i in range(h):                # Parcourt chaque ligne de l'image
        for j in range(w):            # Parcourt chaque colonne de l'image
            region = padded[i:i+sh, j:j+sw]   # Extrait la région de l'image sous le structurant
            # Vérifie si tous les pixels de la région correspondant au structurant sont à 1
            if np.all(region[structure == 0] == 1):
                eroded[i, j] = 0      # Si oui, le pixel central devient 1
    return eroded                    # Retourne l'image érodée

def dilatation(image, structure):
    h, w = image.shape               # Récupère la hauteur et la largeur de l'image
    sh, sw = structure.shape         # Récupère la hauteur et la largeur de l'élément structurant
    pad_h, pad_w = sh // 2, sw // 2  # Calcule le nombre de pixels à ajouter pour le padding
    # Ajoute un padding de zéros autour de l'image pour gérer les bords
    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)
    dilated = np.zeros_like(image)   # Initialise l'image dilatée avec des zéros
    for i in range(h):               # Parcourt chaque ligne de l'image
        for j in range(w):           # Parcourt chaque colonne de l'image
            region = padded[i:i+sh, j:j+sw]   # Extrait la région de l'image sous le structurant
            # Vérifie si au moins un pixel de la région correspondant au structurant est à 1
            if np.any(region[structure == 1] == 1):
                dilated[i, j] = 1    # Si oui, le pixel central devient 1
    return dilated                   # Retourne l'image dilatée

Time_end = []                        # Initialise la liste pour stocker les temps d'exécution
print("Début des tests...")  # Indique le début des tests
# Boucle pour effectuer 300 essais et mesurer le temps d'exécution
for i in range(300):
    time_start = t.time()            # Enregistre le temps de début

    Image = np.random.randint(0, 2, (256, 256))   # Génère une image binaire aléatoire de taille 256x256
    structure = np.ones((4, 4), dtype=np.uint8)   # Définit un élément structurant carré de taille 4x4

    eroded = erosion(Image, structure)            # Applique l'érosion "from scratch"
    opened = dilatation(eroded, structure)  # Applique la dilatation "from scratch"

    Image = Image * opened            # Multiplie l'image originale par l'image ouverte (pour conserver uniquement les pixels ouverts)
    Time_end.append(t.time() - time_start)   # Calcule et stocke le temps d'exécution pour cet essai
    print(f"Essai {i+1}/300 terminé.",flush=True,end="\r")
print("Temps d'exécution moyen : ", np.mean(Time_end), "secondes +- ", np.std(Time_end))
plt.imshow(Image, cmap='gray')
plt.title("Ouverture morphologique")
plt.show()