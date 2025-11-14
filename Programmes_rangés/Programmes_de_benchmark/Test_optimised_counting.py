import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import sys
import time

# Imports des fonctions du package local
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

from Programmes_de_bases.compteur import compteur_particles_optimized

if __name__ == "__main__":
    # Example usage of compteur_particles
    file = r"C:\Users\Félix\Desktop\Programmation\Projet_cea\Particle_tracking_algorithme_project_M1-2-IN\DATA-20251022T080148Z-1-001\DATA\alpha\60sec_alpha_39kbq_2.5cm_r0.t3pa"
     # Replace with your actual file path
    start = time.time()
    t_start = 0.0
    d_time = 100.0
    print("Running optimized counting...")
    results = compteur_particles_optimized(file=file, t_min=t_start, d_time=d_time,progress_bar=True)
    print("Counting Results:", results)
    end = time.time()
    print(f"Optimized counting completed in {end - start:.2f} seconds.")