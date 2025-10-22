import time
from Compteur_v1 import compteur_particles
import numpy as np
def list_files(folder, recursive=False, extensions=None, fullpath=True, include_hidden=False):
    """
    Retourne la liste des fichiers dans `folder`.
    - recursive: parcourir les sous-dossiers si True
    - extensions: None ou liste/tuple de extensions (ex: ['.png', '.npy'])
    - fullpath: si True retourne chemins absolus, sinon noms de fichiers
    - include_hidden: inclure fichiers commençant par '.' si True
    """
    import os

    if not os.path.isdir(folder):
        raise ValueError(f"{folder!r} n'est pas un dossier valide")

    exts = None
    if extensions:
        exts = set(e.lower() if e.startswith('.') else f".{e.lower()}" for e in extensions)

    files = []
    if recursive:
        for root, _, filenames in os.walk(folder):
            for name in filenames:
                if not include_hidden and name.startswith('.'):
                    continue
                if exts and os.path.splitext(name)[1].lower() not in exts:
                    continue
                files.append(os.path.join(root, name) if fullpath else name)
    else:
        for name in sorted(os.listdir(folder)):
            path = os.path.join(folder, name)
            if not os.path.isfile(path):
                continue
            if not include_hidden and name.startswith('.'):
                continue
            if exts and os.path.splitext(name)[1].lower() not in exts:
                continue
            files.append(os.path.abspath(path) if fullpath else name)

    return files

start_time = time.time()
folder = r"C:/Users/Graziani/Desktop/Projet CEA/Particle_tracking_algorithme_project_M1-2-IN/DATA-20251022T080148Z-1-001"
files = list_files(folder, recursive=True, extensions=['.t3pa'])
time_ends = []
for file in files:
    time_start = time.time()
    N_alpha, N_tracks, N_gamma = compteur_particles(file, plot=False)
    time_ends.append(time.time() - time_start)
    print(f"Fichier: {file} numéro {files.index(file)+1}/{len(files)}")
    print(f"  Nombre de particules alpha détectées : {N_alpha}")
    print(f"  Nombre de particules tracks détectées : {N_tracks}")    
    print(f"  Nombre de particules gamma détectées : {N_gamma}")

print(f"Durée totale : {time.time() - start_time} secondes \n Moyenne par fichier : {np.mean(time_ends)} secondes +- {np.std(time_ends)} secondes \n {time_ends}")