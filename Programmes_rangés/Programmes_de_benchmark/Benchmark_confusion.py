# ...existing code...
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import sys
import psutil
import os 



# Imports des fonctions du package local
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

from Programmes_de_bases.compteur import compteur_particles
from Programmes_de_bases.read_file import read
process = psutil.Process(os.getpid())
def list_npy_files(folder: str, recursive: bool = False) -> list:
    """
    Renvoie une liste triée de chemins (str) vers tous les fichiers .npy
    dans le dossier `folder`. Si recursive=True parcourt récursivement.
    """
    p = Path(folder)
    if not p.exists() or not p.is_dir():
        raise FileNotFoundError(f"Dossier introuvable : {folder}")
    files = p.rglob("*.npy") if recursive else p.glob("*.npy")
    
    return sorted( str(f) for f in files)

def image_maker(cluster):
    """Crée une image 256x256 à partir des données de cluster.

    Parameters
    ----------
    cluster : dict
        Dictionnaire contenant les données du cluster avec la clé 'data'.

    Returns
    -------
    numpy.ndarray
        Image 256x256 représentant les données du cluster.
    """
    
    image = np.zeros((256,256))
    for data in cluster:
        image[data[0],data[1]] = 1  # Incrémenter la valeur du pixel
    return image

def get_current_memory_usage_mib():
    # Obtient la mémoire résidente (RSS) en Bytes
    rss_bytes = process.memory_info().rss
    # Convertit en MegaBytes (MiB)
    return rss_bytes / (1024 * 1024)


if __name__ == "__main__":
    cluster_files =[np.load(i,allow_pickle=True) for i in list_npy_files("C:/Users/Félix/Desktop/Programmation/Projet_cea/Particle_tracking_algorithme_project_M1-2-IN/DATA-20251022T080148Z-1-001/DATA/Combined_Am_SrY/2.5cm", recursive=False)]

    N_alpha_true = 0
    N_electron_true = 0
    N_gamma_true = 0
    N_muon_true = 0
    N_other_true = 0

    N_alpha_algo = 0
    N_electron_algo = 0
    N_gamma_algo = 0
    N_muon_algo = 0


    for clusters in cluster_files:
    
        for i,cluster in enumerate(clusters[2:]):
            print(f"cluster {i}/{len(clusters)-2}", end='\r', flush=True)
    
            #Traitement des vraies valeurs
            N_alpha_true += cluster["counts"]["alpha"]
            N_electron_true += cluster["counts"]["electron"]
            N_gamma_true += cluster["counts"]["gamma"]  
            N_muon_true += cluster["counts"]["muon"]    
            N_other_true += cluster["counts"]["other"]

            #Traitement des valeurs algorithmiques
            counts = compteur_particles(image_maker(cluster["pixels"]),is_slice=True)["Counts"]
            N_alpha_algo_i,N_electron_algo_i,N_muon_algo_i,N_gamma_algo_i = counts["alpha"],counts["electrons"],counts["muons"],counts["gamma"]
            N_alpha_algo += N_alpha_algo_i
            N_electron_algo += N_electron_algo_i
            N_gamma_algo += N_gamma_algo_i
            N_muon_algo += N_muon_algo_i    
            
    print("\nRÉSULTATS DE LA CONFUSION :")
    # Relative errors (algo - truth) / truth, expressed both as counts and percent
    def rel_err(algo, truth):
        try:
            if truth == 0:
                return None
            return (algo - truth) / truth
        except Exception:
            return None

    print('\nERREURS RELATIVES :')
    items = [
        ('Alpha', N_alpha_algo, N_alpha_true),
        ('Electrons', N_electron_algo, N_electron_true),
        ('Muons', N_muon_algo, N_muon_true),
        ('Gamma', N_gamma_algo, N_gamma_true),
    ]

    total_true = N_alpha_true + N_electron_true + N_muon_true + N_gamma_true
    total_algo = N_alpha_algo + N_electron_algo + N_muon_algo + N_gamma_algo

    for name, a, t in items:
        r = rel_err(a, t)
        if r is None:
            if t == 0 and a == 0:
                print(f"{name}: truth=0, algo=0 → relative error: 0.00 (0%)")
            elif t == 0:
                print(f"{name}: truth=0, algo={a} → relative error: undefined (division by zero)")
            else:
                print(f"{name}: relative error unavailable")
        else:
            print(f"{name}: algo={a}, truth={t}, rel_err={r:.4f} ({abs(r)*100:.2f}%)")

    # Totals
    r_tot = rel_err(total_algo, total_true)
    if r_tot is None:
        if total_true == 0 and total_algo == 0:
            print(f"Total: truth=0, algo=0 → relative error: 0.00 (0%)")
        elif total_true == 0:
            print(f"Total: truth=0, algo={total_algo} → relative error: undefined (division by zero)")
    else:
        print(f"Total: algo={total_algo}, truth={total_true}, rel_err={r_tot:.4f} ({r_tot*100:.2f}%)")

    