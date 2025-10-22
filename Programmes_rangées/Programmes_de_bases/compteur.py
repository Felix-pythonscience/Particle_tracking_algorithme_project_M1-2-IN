import numpy as np
from read_file import read,slice,slice_Tot
from scipy.ndimage import label
from filtres import filtre_alpha, filtre_tracks
from plot_results import plot_results

def compteur_particles(file = "None", t= 0, d_time = None,plot = False):
    """Count particle types in a time window and optionally plot the results.

    This function reads the data (or accepts an already-loaded DataFrame/array),
    builds a binary image using `slice`, applies `filtre_alpha` and `filtre_tracks` to
    separate alpha, track and gamma components, then returns the number of connected
    components found in each category.

    Parameters
    ----------
    file : str or pandas.DataFrame, optional
        Path to the input file or an already loaded DataFrame/array. When a string
        is passed it will be read with `read()`.
    t : float, optional
        Start time of the analysis window (currently unused; kept for API parity).
    d_time : float, optional
        Duration of the time window. If None, defaults to max(time)/100.
    plot : bool, optional
        If True, display diagnostic plots using `plot_results`.

    Returns
    -------
    tuple
        (N_alpha, N_tracks, N_gamma) counts of connected components for each class.
    """
    data = file if not(type(file) == str) else read(file)

    d_time = d_time if not d_time==None else max(data.iloc[:, 1]) / 100  # Diviser le temps

    image = slice(data.to_numpy(), 0, d_time)

    image_without_alpha, image_alpha = filtre_alpha(image)# Appliquer le filtre pour enlever les tracks

    image_without_tracks, image_tracks = filtre_tracks(image_without_alpha)# Appliquer le filtre pour enlever les tracks

    N_alpha = label(image_alpha)[1]
    N_tracks = label(image_tracks)[1]   
    N_gamma = label(image_without_tracks)[1]
    if plot:
        image_couleur = slice_Tot(data.to_numpy(), 0, d_time) # Image colorier par le TOT pour visualisation
        plot_results(image, image_alpha, image_tracks, image_without_tracks, image_couleur)

    return N_alpha, N_tracks, N_gamma
if __name__ == "__main__":
    # Lecture des données et création de l'image binaire
    #file = "C:/Users/Graziani/Desktop/Projet CEA/Particle_tracking_algorithme_project_M1-2-IN/DATA-20251022T080148Z-1-001/DATA/alpha/60sec_alpha_39kbq_2.5cm_r0.t3pa"
    file = "C:/Users/Graziani/Desktop/Projet CEA/Particle_tracking_algorithme_project_M1-2-IN/DATA-20251022T080148Z-1-001/DATA/Combined_Am_SrY/2.5cm/2.5cm_r0.t3pa"
    
    N_alpha, N_tracks, N_gamma = compteur_particles(file =file, plot=True)

    print(f"Nombre de particules alpha détectées : {N_alpha}")
    print(f"Nombre de particules tracks détectées : {N_tracks}")    
    print(f"Nombre de particules gamma détectées : {N_gamma}")
