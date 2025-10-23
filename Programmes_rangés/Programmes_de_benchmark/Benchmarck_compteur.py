import time
from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt
# Ajoute le dossier parent du fichier courant au chemin Python
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))
from Programmes_de_bases.compteur import compteur_particles
from Programmes_de_bases.read_file import read


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
script_dir = Path(__file__).resolve().parent
folder = r"C:/Users/Félix/Desktop/Programmation/Projet_cea/Particle_tracking_algorithme_project_M1-2-IN/DATA-20251022T080148Z-1-001/DATA/alpha"
files = list_files(folder, recursive=True, extensions=['.t3pa'])
time_ends = []
N_alpha_total = []
N_tracks_total = []
N_gamma_total = []
x = np.linspace(1, 2000, 100, dtype=int)  # dt en ms
dts =[]
for file in files:
    data = read(file)
    time_max = max(data.iloc[:, 1])
    d_time = time_max / x  # Diviser le temps
    for i, dt in enumerate(d_time):
        
        time_start = time.time()
        N_alpha, N_tracks, N_gamma = 0, 0, 0

        for t in range(x[i]):
            print(f" Traitement du temps {t+1}/{x[i]}", end='\r', flush=True)
            N_alpha_dt, N_tracks_dt, N_gamma_dt = compteur_particles(file=data, t=t * dt, d_time=dt,
                                                                     save=[True if t == 0 else False,
                                                                           Path(f"dt = 1 divisé par {str(x[i]).zfill(4)}"),
                                                                           script_dir / "Benchmark_Results" / "compteur2" / "Evolution_détections_en_fonction_de_dt"])
            N_alpha += N_alpha_dt
            N_tracks += N_tracks_dt
            N_gamma += N_gamma_dt
        time_ends.append(time.time() - time_start)
        print(f"Fichier: {file} numéro {files.index(file)+1}/{len(files)}")
        print(f"  Nombre de particules alpha détectées : {N_alpha}")
        print(f"  Nombre de particules tracks détectées : {N_tracks}")    
        print(f"  Nombre de particules gamma détectées : {N_gamma}")
        N_alpha_total.append(N_alpha)
        N_tracks_total.append(N_tracks) 
        N_gamma_total.append(N_gamma)
        dts.append(dt/time_max)


fig, axes = plt.subplots(2, 2, figsize=(12, 8))
ax1, ax2, ax3, ax4 = axes.ravel()

ax1.plot(dts, N_alpha_total, marker='o')
ax1.set_title('Alpha')
ax1.set_xlabel('dt')
ax1.set_ylabel('Nombre de particules')
ax1.set_xscale('log')
ax2.plot(dts, N_tracks_total, marker='o', color='C1')
ax2.set_title('Tracks')
ax2.set_xlabel('dt')
ax2.set_ylabel('Nombre de particules')
ax2.set_xscale('log')

ax3.plot(dts, N_gamma_total, marker='o', color='C2')
ax3.set_title('Gamma')
ax3.set_xlabel('dt')
ax3.set_ylabel('Nombre de particules')
ax3.set_xscale('log')

ax4.plot(dts, np.array(N_alpha_total) + np.array(N_tracks_total) + np.array(N_gamma_total), 
         marker='o', color='k')
ax4.set_title('Total')
ax4.set_xlabel('dt ')
ax4.set_ylabel('Nombre de particules')
ax4.set_xscale('log')
fig.tight_layout()


outdir = script_dir / "Benchmark_Results" / "compteur2"
outdir.mkdir(parents=True, exist_ok=True)
fig.savefig(outdir / "N_detections_en_fonction_de_dt.png", dpi=300, bbox_inches='tight')

plt.show(block=False)

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