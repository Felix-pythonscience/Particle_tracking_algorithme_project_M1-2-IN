import numpy as np
from pathlib import Path
import sys
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))
try:
    # allow running as part of a package (preferred)
    from .read_file import read, slice, slice_Tot
    from .filtres import filtre_alpha, filtre_tracks
    from .plot_results import plot_results
    from .event_detector_v4 import event_counting_alpha, event_counting_electron_muon,event_counting_photon
    # diagnostic: indicate which import branch succeeded
   
except Exception:
    # fallback when the module is executed directly as a script
    # attempt to import from the same directory
    from read_file import read, slice, slice_Tot
    from filtres import filtre_alpha, filtre_tracks
    from plot_results import plot_results
    from event_detector_v4 import event_counting_alpha, event_counting_electron_muon,event_counting_photon
    # diagnostic: indicate fallback import used
   


def compteur_particles(file = "None", t= 0, d_time = None, plot = False, block = False,
                    return_images=False, is_slice = False,save = [False,"plot_results.png",Path.cwd()],
                    discrimination_criteria = {"alpha":{}, "electron_muon":{}},
                    loop=0):
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
        Start time of the analysis window .
    d_time : float, optional
        Duration of the time window. If None, defaults to 50.
    plot : bool, optional
        If True, display diagnostic plots using `plot_results`.
    

    return_images : bool, optional
        If True, return the 3 splited images + the original image in the results dictionary.
    is_slice : bool, optional
        If True, the input `file` argument is already a sliced image (i.e. the result
        of the `slice` function) and should be used directly.
    save : list, optional
        A list where the first element is a boolean indicating whether to save the results or not,
        the second element is the last folder for the saveds plots, and the third element
        is the path where the results should be saved.
    discrimination_criteria : dict, optional
        Dictionary containing discrimination criteria for cluster based filtering.

    Returns
    -------
    dict
        Dictionary containing counts and optionally images.
    """
    if is_slice:
        image = file
    else:
        data = file if not(type(file) == str) else read(file)
        d_time = d_time if d_time!=None else 150000000  # Diviser le temps

        image = slice(data.to_numpy(), t, d_time)

    # Morphological filtering of alpha
    image_without_alpha, image_alpha = filtre_alpha(image)# Appliquer le filtre pour enlever les tracks

    
    #Counting particles + cluster based filtering of alpha

    N_alpha,falses_alphas = event_counting_alpha(image_alpha, solidity_threshold=discrimination_criteria["alpha"].get("solidity_threshold", 0.2))

    #filtering of tracks
    # Appliquer le filtre pour enlever les tracks
    image_gamma, image_tracks = filtre_tracks(image_without_alpha,falses_alphas=falses_alphas)


    N_electrons , N_muons, N_alpha_corr = event_counting_electron_muon(electron_muon_matrix = image_tracks,
                                                                       eccentricity_threshold=discrimination_criteria["electron_muon"].get("eccentricity_threshold", 0.99),
                                                                       solidity_threshold=discrimination_criteria["electron_muon"].get("solidity_threshold", 0.75),
                                                                       area_threshold=discrimination_criteria["electron_muon"].get("area_threshold", 10))

    N_alpha += N_alpha_corr

    N_gamma = event_counting_photon(image_gamma)


    results = {"Counts": {
                "alpha": N_alpha,
                "electrons": N_electrons,
                "muons": N_muons,
                "gamma": N_gamma
            },
            "Images": {
                "original": image,
                "alpha": image_alpha,
                "tracks": image_tracks,
                "gamma": image_gamma
            } if return_images else None
            }

    if plot:

        image_couleur = slice_Tot(data.to_numpy(), t, d_time) # Image coloriée par le TOT pour visualisation
        plot_results(image, image_alpha, image_tracks, image_gamma, image_couleur, block = block, save=save)

    if save[0]:
        outdir = Path(save[2]) / Path(save[1])
        outdir.mkdir(parents=True, exist_ok=True)
        np.save(outdir / "image_originale.npy", image.astype(np.uint8), allow_pickle=False)
        np.save(outdir / "image_alpha.npy", image_alpha.astype(np.uint8), allow_pickle=False)
        np.save(outdir / "image_tracks.npy", image_tracks.astype(np.uint8), allow_pickle=False)
        np.save(outdir / "image_gamma.npy", image_gamma.astype(np.uint8), allow_pickle=False)


    return results



if __name__ == "__main__":
    # Lecture des données et création de l'image binaire
    #file = "C:/Users/Graziani/Desktop/Projet CEA/Particle_tracking_algorithme_project_M1-2-IN/DATA-20251022T080148Z-1-001/DATA/alpha/60sec_alpha_39kbq_2.5cm_r0.t3pa"
    #file = "C:/Users/Graziani/Desktop/Projet_CEA/Projet/Particle_tracking_algorithme_project_M1-2-IN/DATA-20251022T080148Z-1-001/DATA/Combined_Am_SrY/2.5cm/2.5cm_r0.t3pa"
    file = "C:/Users/Graziani/Desktop/Projet_CEA/Projet/Particle_tracking_algorithme_project_M1-2-IN/DATA-20251022T080148Z-1-001/DATA/alpha/60sec_alpha_39kbq_2.5cm_r0.t3pa"
    #Counting particles
    data = read(file)
    dt = 15E6
    n_windows = int(np.ceil(data.iloc[:, 1].max() / dt)) if dt > 0 else 1
    N_alpha_total = 0
    N_electrons_total = 0      
    N_muons_total = 0
    N_gamma_total = 0
    for i in range(n_windows):
        counts = compteur_particles(file, t=i*dt, d_time=dt, plot=False, block=False, save=[False,"Test_alpha_qui_passe_pour_dafuk",Path.cwd()])
        #spliting results
        N_alpha, N_electrons, N_muons, N_gamma = counts["Counts"]["alpha"], counts["Counts"]["electrons"], counts["Counts"]["muons"], counts["Counts"]["gamma"]
        N_alpha_total += N_alpha
        N_electrons_total += N_electrons
        N_muons_total += N_muons
        N_gamma_total += N_gamma
        print(f"Fenêtre {i+1}/{n_windows} : Alpha={N_alpha}, Electrons={N_electrons}, Muons={N_muons}, Gamma={N_gamma}", end='\r', flush=True)
    print("\nRésultats totaux sur toutes les fenêtres temporelles :")
    print(f"Nombre de particules alpha détectées : {N_alpha_total}")
    print(f"Nombre de particules électrons détectées : {N_electrons_total}")
    print(f"Nombre de particules muons détectées : {N_muons_total}")
    print(f"Nombre de particules gamma détectées : {N_gamma_total}")