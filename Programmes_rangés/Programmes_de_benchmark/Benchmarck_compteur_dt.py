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
                # Ajouter chemin absolu ou nom selon fullpath
                files.append(os.path.join(root, name) if fullpath else name)
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
            files.append(os.path.abspath(path) if fullpath else name)

    return files
start_time = time.time()  # moment de départ pour la mesure globale
script_dir = Path(__file__).resolve().parent  # dossier contenant ce script

# Dossier contenant les fichiers .t3pa à traiter (modifiable)
folder = r"C:/Users/Félix/Desktop/Programmation/Projet_cea/Particle_tracking_algorithme_project_M1-2-IN/DATA-20251022T080148Z-1-001/DATA/combined_Am_SrY/2.5cm"
# Récupère la liste des fichiers .t3pa sous ce dossier (récursif)
files = list_files(folder, recursive=True, extensions=['.t3pa'])

# Listes pour stocker les résultats et temps
time_ends = []
N_alpha_total = []
N_tracks_total = []
N_gamma_total = []

# x contient les différentes résolutions temporelles testées (nombre de sous-intervalles)
x = np.linspace(100, 2000, 100, dtype=int)  # dt en ms (array de 100 valeurs entières entre 100 et 2000)
dts = []  # stocke les dt normalisés par la durée totale

# Boucle sur les fichiers (ici prise d'exemple sur le premier fichier seulement)
for file in [files[0]]:
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
                t=t * dt,  # temps de départ de la tranche
                d_time=dt,  # durée de la tranche
                save=[True if t == 0 else False,  # sauvegarde seulement pour la première tranche
                      Path(f"dt = 1 divisé par {str(x[i]).zfill(4)}"),
                      script_dir / "Benchmark_Results" / "compteur5" / "Evolution_détections_en_fonction_de_dt"])
            # Accumule les résultats
            N_alpha += N_alpha_dt
            N_tracks += N_tracks_dt
            N_gamma += N_gamma_dt
        # Mesure le temps écoulé pour ce fichier et ce dt
        time_ends.append(time.time() - time_start)
        # Affiche un résumé pour l'utilisateur
        print(f"Fichier: {file} numéro {files.index(file)+1}/{len(files)}")
        print(f"  Nombre de particules alpha détectées : {N_alpha}")
        print(f"  Nombre de particules tracks détectées : {N_tracks}")
        print(f"  Nombre de particules gamma détectées : {N_gamma}")
        # Stocke les résultats pour tracés ultérieurs
        N_alpha_total.append(N_alpha)
        N_tracks_total.append(N_tracks)
        N_gamma_total.append(N_gamma)
        dts.append(dt / time_max)


fig, axes = plt.subplots(2, 2, figsize=(12, 8))  # création d'une figure 2x2 pour les 4 courbes
ax1, ax2, ax3, ax4 = axes.ravel()  # aplatit les axes en une liste pour affectation facile

# Trace la courbe du nombre d'alpha en fonction de dt
ax1.plot(dts, N_alpha_total, marker='o')
ax1.set_title('Alpha')
ax1.set_xlabel('dt')
ax1.set_ylabel('Nombre de particules')
ax1.set_xscale('log')

# Trace la courbe des tracks
ax2.plot(dts, N_tracks_total, marker='o', color='C1')
ax2.set_title('Tracks')
ax2.set_xlabel('dt')
ax2.set_ylabel('Nombre de particules')
ax2.set_xscale('log')

# Trace la courbe des gamma
ax3.plot(dts, N_gamma_total, marker='o', color='C2')
ax3.set_title('Gamma')
ax3.set_xlabel('dt')
ax3.set_ylabel('Nombre de particules')
ax3.set_xscale('log')

# Trace le total (somme des trois catégories)
ax4.plot(dts, np.array(N_alpha_total) + np.array(N_tracks_total) + np.array(N_gamma_total), 
         marker='o', color='k')
ax4.set_title('Total')
ax4.set_xlabel('dt ')
ax4.set_ylabel('Nombre de particules')
ax4.set_xscale('log')
fig.tight_layout()


# Prépare le dossier de sortie et enregistre la figure principale
outdir = script_dir / "Benchmark_Results" / "compteur4"
outdir.mkdir(parents=True, exist_ok=True)
fig.savefig(outdir / "N_detections_en_fonction_de_dt.png", dpi=300, bbox_inches='tight')

plt.show(block=False)  # affiche la figure sans bloquer l'exécution

# Figure pour les temps d'exécution
fig2, ax_time = plt.subplots(figsize=(8, 4))
print(f"Durée totale : {time.time() - start_time} secondes \n Moyenne par fichier : {np.mean(time_ends)} secondes +- {np.std(time_ends)} secondes \n {time_ends}")
ax_time.plot(dts, time_ends, marker='o', label='Temps de calcul du fichier en fonction de dt')
ax_time.set_xlabel('dt')
ax_time.set_ylabel('Temps (s)')
ax_time.set_xscale('log')
ax_time.legend()
plt.tight_layout()

outdir.mkdir(parents=True, exist_ok=True)
fig2.savefig(outdir / "Temps_dExecution_en_fonction_de_dt.png", dpi=300, bbox_inches='tight')
plt.show()