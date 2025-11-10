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
    # Load cluster files once
    cluster_files = [np.load(i, allow_pickle=True) for i in list_npy_files(
        "C:/Users/Félix/Desktop/Programmation/Projet_cea/Particle_tracking_algorithme_project_M1-2-IN/DATA-20251022T080148Z-1-001/DATA/Combined_Am_SrY/2.5cm",
        recursive=False
    )]

    # First: compute ground-truth totals from saved cluster files (counts stored per-cluster)
    N_alpha_true = 0
    N_electron_true = 0
    N_gamma_true = 0
    N_muon_true = 0
    N_other_true = 0

    for clusters in cluster_files:
        for cluster in clusters[2:]:
            N_alpha_true += int(cluster["counts"].get("alpha", 0))
            N_electron_true += int(cluster["counts"].get("electron", 0))
            N_gamma_true += int(cluster["counts"].get("gamma", 0))
            N_muon_true += int(cluster["counts"].get("muon", 0))
            N_other_true += int(cluster["counts"].get("other", 0))

    print("Ground-truth totals:")
    print(f"Alpha: {N_alpha_true}, Electrons: {N_electron_true}, Muons: {N_muon_true}, Gamma: {N_gamma_true}, Other: {N_other_true}")

    # Prepare solidity threshold sweep
    solidity_values = np.round(np.arange(0.6, 0.991, 0.02), 3)
    rel_err_alpha = []
    rel_err_electron = []

    # Setup live plot (we'll plot absolute relative error in percent)
    plt.ion()
    fig, ax = plt.subplots(figsize=(8, 4))
    line_alpha, = ax.plot([], [], label='alpha abs rel err (%)')
    line_elec, = ax.plot([], [], label='electron abs rel err (%)')
    ax.set_xlabel('solidity threshold')
    ax.set_ylabel('relative error (%)')
    ax.set_ylim(-100, 100)
    ax.grid(True)
    ax.legend()

    # helper for relative error
    def rel_err(algo, truth):
        if truth == 0:
            return None
        return (algo - truth) / truth

    # Precompute total number of clusters for progress reporting
    total_clusters = sum((len(clusters) - 2) for clusters in cluster_files)

    # Sweep thresholds; for each threshold compute algorithm totals and update the plot
    for idx, sol in enumerate(solidity_values):
        print(f"Threshold {idx+1}/{len(solidity_values)}: solidity={sol}")
        N_alpha_algo = 0
        N_electron_algo = 0
        N_gamma_algo = 0
        N_muon_algo = 0

        # For each stored cluster, rebuild an image and run the algorithm with current solidity
        processed = 0
        for clusters in cluster_files:
            for cluster in clusters[2:]:
                processed += 1
                print(f"Processing cluster {processed}/{total_clusters} (solidity={sol})", end='\r', flush=True)
                # build image from pixel coords
                img = image_maker(cluster["pixels"])  # 256x256
                counts = compteur_particles(
                    img,
                    is_slice=True,
                    discrimination_criteria={
                        "alpha": {},
                        "electron_muon": {
                            "eccentricity_threshold": 0.99,
                            "solidity_threshold": float(sol),
                            "area_threshold": 10,
                        },
                    },
                )["Counts"]

                N_alpha_algo += int(counts.get("alpha", 0))
                # note: compteur_particles returns electrons under key 'electrons'
                N_electron_algo += int(counts.get("electrons", 0))
                N_gamma_algo += int(counts.get("gamma", 0))
                N_muon_algo += int(counts.get("muons", 0))

        # compute absolute relative errors in percent for alpha and electrons, handle zero-truth
        ra = rel_err(N_alpha_algo, N_alpha_true)
        re = rel_err(N_electron_algo, N_electron_true)
        # convert to absolute percent (NaN where undefined)
        val_a = 0.0 if (ra is None and N_alpha_algo == 0) else (ra * 100.0 if ra is not None else np.nan)
        val_e = 0.0 if (re is None and N_electron_algo == 0) else (re * 100.0 if re is not None else np.nan)
        rel_err_alpha.append(val_a)
        rel_err_electron.append(val_e)

        # update plot
        xs = solidity_values[: idx + 1]
        line_alpha.set_data(xs, rel_err_alpha)
        line_elec.set_data(xs, rel_err_electron)
        ax.set_xlim(solidity_values[0], solidity_values[-1])
        # autoscale y for percent values (>=0); keep reasonable limits
        all_vals = [v for v in rel_err_alpha + rel_err_electron if not (v is None or np.isnan(v))]
        if all_vals:
            ymin = min(-10.0, min(all_vals) * 1.1)
            ymax = max(10.0, max(all_vals) * 1.1)
            ax.set_ylim(ymin, ymax)
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(0.01)

    # Finalize plot
    plt.ioff()
    ax.set_title('Absolute relative error vs solidity threshold (%)')
    fig.tight_layout()
    fig.savefig('relative_error_vs_solidity.png')
    print('Sweep complete, plot saved to relative_error_vs_solidity.png')

    # Print final relative errors (absolute percent)
    print('\nERREURS RELATIVES (valeurs absolues en %) :')
    items = [
        ('Alpha', N_alpha_algo, N_alpha_true),
        ('Electrons', N_electron_algo, N_electron_true),
        ('Muons', N_muon_algo, N_muon_true),
        ('Gamma', N_gamma_algo, N_gamma_true),
    ]

    total_true = N_alpha_true + N_electron_true + N_muon_true + N_gamma_true
    total_algo = N_alpha_algo + N_electron_algo + N_muon_algo + N_gamma_algo

    for name, a, t in items:
        if t == 0:
            if a == 0:
                print(f"{name}: truth=0, algo=0 → abs relative error: 0.00%")
            else:
                print(f"{name}: truth=0, algo={a} → abs relative error: undefined (division by zero)")
        else:
            r = abs((a - t) / t) * 100.0
            print(f"{name}: algo={a}, truth={t}, abs_rel_err={r:.2f}%")

    # Totals
    if total_true == 0:
        if total_algo == 0:
            print("Total: truth=0, algo=0 → abs relative error: 0.00%")
        else:
            print(f"Total: truth=0, algo={total_algo} → abs relative error: undefined (division by zero)")
    else:
        r_tot = abs((total_algo - total_true) / total_true) * 100.0
        print(f"Total: algo={total_algo}, truth={total_true}, abs_rel_err={r_tot:.2f}%")

    