"""
Créé le 23/10/2025 par félix graziani pour rechercher une trace de muon dans le background

"""



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
                # Filtrer par extension si demandé
                if exts and os.path.splitext(name)[1].lower() not in exts:
                    continue
                # Construire un Path et renvoyer un Path résolu si fullpath True
                p = Path(os.path.join(root, name))
                files.append(p.resolve() if fullpath else Path(name))
    else:
        # Parcours non récursif, tri alphabétique
        for name in sorted(os.listdir(folder)):
            path = os.path.join(folder, name)
            # Ne garder que les fichiers (pas les dossiers)
            if not os.path.isfile(path):
                continue
            if not include_hidden and name.startswith('.'):
                continue
            if exts and os.path.splitext(name)[1].lower() not in exts:
                continue
            p = Path(os.path.abspath(path))
            files.append(p.resolve() if fullpath else Path(name))

    return files
start_time = time.time()  # moment de départ pour la mesure globale
script_dir = Path(__file__).resolve().parent  # dossier contenant ce script

# Dossier contenant les fichiers .t3pa à traiter (modifiable)
folder = r"C:/Users/Félix/Desktop/Programmation/Projet_cea/Particle_tracking_algorithme_project_M1-2-IN/DATA-20251022T080148Z-1-001/DATA/bkg"
# Récupère la liste des fichiers .t3pa sous ce dossier (récursif)
files = list_files(folder, recursive=True, extensions=['.t3pa'])

# Listes pour stocker les résultats et temps
time_ends = []
N_alpha_total = []
N_tracks_total = []
N_gamma_total = []

# x contient les différentes résolutions temporelles testées (nombre de sous-intervalles)
x = np.array([5])  # diviseurs de dt 
dts = []  # stocke les dt normalisés par la durée totale
print(files)
# Boucle sur les fichiers (ici prise d'exemple sur le premier fichier seulement)
for file in files:
    # Lecture du fichier en DataFrame (fonction read du module local)
    data = read(file)
    # Temps maximal présent dans les données (colonne 1)
    time_max = max(data.iloc[:, 1])
    # d_time correspond à la liste des durées pour chaque résolution testée
    d_time = time_max / x  # Diviser le temps en x intervalles
    for i, dt in enumerate(d_time):
        # Début du chronométrage pour cette valeur de dt
        time_start = time.time()
        # compteurs cumulés pour ce fichier et ce dt
        N_alpha, N_tracks, N_gamma = 0, 0, 0

        # Traitement des x[i] fenêtres temporelles pour cette résolution
        for t in range(x[i]):
            # Affiche une progression sur la même ligne
            print(f" Traitement du temps {t+1}/{x[i]}", end='\r', flush=True)
            # Appel de la fonction de comptage sur la tranche courante
            N_alpha_dt, N_tracks_dt, N_gamma_dt = compteur_particles(
                file=data,  # on passe l'objet DataFrame directement pour éviter une relecture disque
                t= t * dt,  # temps de départ de la tranche
                d_time=dt,  # durée de la tranche
                plot=True,
                block=True,
                save=[True,  # sauvegarde seulement pour la première tranche
                Path(f"t = {t}"),
                script_dir / "Benchmark_Results" / "muon_search" / Path(file).stem
                ]
            )
            print(t)
