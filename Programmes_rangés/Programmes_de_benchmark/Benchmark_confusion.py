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

print(f"Mémoire de départ : {get_current_memory_usage_mib():.2f} MiB")

if __name__ == "__main__":
    cluster_files =[np.load(i,allow_pickle=True) for i in list_npy_files("C:/Users/Graziani/Desktop/Projet_CEA/Projet/Particle_tracking_algorithme_project_M1-2-IN/DATA-20251022T080148Z-1-001/DATA/Combined_Am_SrY/2.5cm", recursive=False)]

    N_alpha_true = 0
    N_electron_true = 0
    N_gamma_true = 0
    N_muon_true = 0
    N_other_true = 0

    N_alpha_algo = 0
    N_electron_algo = 0
    N_gamma_algo = 0
    N_muon_algo = 0


    print("HAHAHAHAHAH")
    for clusters in cluster_files:
    
        for i,cluster in enumerate(clusters[2:]):
            print(f"cluster {i}/{len(clusters)-2}")
            print(f"Mémoire utilisée : {get_current_memory_usage_mib():.2f} MiB")
            #Traitement des vraies valeurs
            N_alpha_true += cluster["counts"]["alpha"]
            N_electron_true += cluster["counts"]["electron"]
            N_gamma_true += cluster["counts"]["gamma"]  
            N_muon_true += cluster["counts"]["muon"]    
            N_other_true += cluster["counts"]["other"]

            #Traitement des valeurs algorithmiques
            N_alpha_algo_i,N_electron_algo_i,N_gamma_algo_i,N_muon_algo_i = compteur_particles(image_maker(cluster["pixels"]),slice=True)
            N_alpha_algo += N_alpha_algo_i
            N_electron_algo += N_electron_algo_i
            N_gamma_algo += N_gamma_algo_i
            N_muon_algo += N_muon_algo_i    
            

    print("VRAIES VALEURS :")
    print(f"Nombre de particules alpha détectées : {N_alpha_true}")     
    print(f"Nombre de particules électrons détectées : {N_electron_true}")
    print(f"Nombre de particules muons détectées : {N_muon_true}")

    print(f"Nombre de particules gamma détectées : {N_gamma_true}")
    print(f"Nombre de particules autres détectées : {N_other_true}")
    print("\n")
    print("VALEURS ALGORITHMIQUES :")
    print(f"Nombre de particules alpha détectées : {N_alpha_algo}")     
    print(f"Nombre de particules électrons détectées : {N_electron_algo}")
    print(f"Nombre de particules muons détectées : {N_muon_algo}")
    print(f"Nombre de particules gamma détectées : {N_gamma_algo}")

