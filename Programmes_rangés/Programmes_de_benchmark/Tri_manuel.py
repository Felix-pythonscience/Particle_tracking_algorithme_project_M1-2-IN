import time  # pour mesurer les durées d'exécution
from pathlib import Path  # gestion de chemins OS-indépendants
import sys  # manipulation du sys.path pour importer depuis le repo
import numpy as np  # calcul numérique
import matplotlib.pyplot as plt  # tracés

# Ajoute le dossier parent du fichier courant au chemin Python afin de pouvoir
# importer les modules locaux (Programmes_de_bases) même si le script est lancé
# directement depuis ce dossier.
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

# Imports des fonctions du package local
from Programmes_de_bases.compteur import compteur_particles
from Programmes_de_bases.read_file import read


def list_files(folder, recursive=False, extensions=None, fullpath=True, include_hidden=False):
    """Retourne la liste des fichiers dans `folder`.

    Paramètres
    ----------
    folder : str
        chemin du dossier à lister
    recursive : bool
        si True, parcourt récursivement les sous-dossiers
    extensions : list or tuple or None
        filtre par extensions, ex: ['.t3pa']
    fullpath : bool
        si True retourne chemins absolus, sinon noms seuls
    include_hidden : bool
        si True inclut les fichiers commençant par '.'

    Retour
    -----
    list
        liste de chemins/noms de fichiers
    """
    import os

    # Vérifie que le dossier est valide
    if not os.path.isdir(folder):
        raise ValueError(f"{folder!r} n'est pas un dossier valide")

    # Normalise les extensions demandées (avec un point et en minuscule)
    exts = None
    if extensions:
        exts = set(e.lower() if e.startswith('.') else f".{e.lower()}" for e in extensions)

    files = []
    if recursive:
        # Parcours récursif du dossier
        for root, _, filenames in os.walk(folder):
            for name in filenames:
                # Ignorer les fichiers cachés si demandé
                if not include_hidden and name.startswith('.'):
                    continue
                # Vérifie l'extension si spécifiée
                if exts and not name.lower().endswith(tuple(exts)):
                    continue
                # Ajoute le chemin complet ou le nom selon le paramètre
                files.append(os.path.join(root, name) if fullpath else name)
    else:
        # Parcours non récursif du dossier
        for name in os.listdir(folder):
            # Ignorer les fichiers cachés si demandé
            if not include_hidden and name.startswith('.'):
                continue
            # Vérifie l'extension si spécifiée
            if exts and not name.lower().endswith(tuple(exts)):
                continue
            # Ajoute le chemin complet ou le nom selon le paramètre
            files.append(os.path.join(folder, name) if fullpath else name)

    return files
if __name__ == "__main__":
    # Dossier contenant les fichiers à analyser
    data_folder = "C:/Users/Graziani/Desktop/Projet CEA/Particle_tracking_algorithme_project_M1-2-IN/DATA-20251022T080148Z-1-001/DATA/Combined_Am_SrY/2.5cm/"

    # Liste des fichiers .t3pa dans le dossier (non récursif)
    files = list_files(data_folder, extensions=['.t3pa'], recursive=False, fullpath=True)

    # Initialisation des listes pour stocker les résultats
    alpha_counts = []
    track_counts = []
    gamma_counts = []
    processing_times = []

    # Boucle sur chaque fichier pour effectuer le comptage
    for file in files:
 

        Data = [compteur_particles(file=file, plot=False,images_debug=True)]
        images = Data[3:-1]
        
    



