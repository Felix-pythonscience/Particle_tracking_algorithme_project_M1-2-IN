import numpy as np
from pathlib import Path
import sys
import time
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))
try:
    # allow running as part of a package (preferred)
    from .read_file import read, slice, slice_Tot,optimised_slice
    from .filtres import filtre_alpha, filtre_tracks
    from .plot_results import plot_results
    from .event_detector_v6 import event_counting_alpha, event_counting_electron_muon,event_counting_photon
    # diagnostic: indicate which import branch succeeded
   
except Exception:
    # fallback when the module is executed directly as a script
    # attempt to import from the same directory
    from read_file import read, slice, slice_Tot,optimised_slice
    from filtres import filtre_alpha, filtre_tracks
    from plot_results import plot_results
    from event_detector_v6 import event_counting_alpha, event_counting_electron_muon,event_counting_photon
    # diagnostic: indicate fallback import used
   


def compteur_particles(file = "None", t= 0, d_time = None, plot = False, block = False,
                    return_images=False, is_slice = False,save = [False,"plot_results.png",Path.cwd()],
                    discrimination_criteria = {"electron_muon":{}},
                    ):
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
        d_time = d_time if d_time!=None else 150  # Diviser le temps

        image = slice(data.to_numpy(), t, d_time)

    
    # Morphological filtering of alpha
    image_without_alpha, image_alpha = filtre_alpha(image)# Appliquer le filtre pour enlever les tracks

    
    #Counting particles + cluster based filtering of alpha

    N_alpha = event_counting_alpha(image_alpha)
    #filtering of tracks
    # Appliquer le filtre pour enlever les tracks
    image_gamma, image_tracks = filtre_tracks(image_without_alpha)


    N_electrons , N_muons, N_alpha_corr, Muon_matrix , Alpha_correction_matrix = event_counting_electron_muon(
                                                                       electron_muon_matrix=image_tracks,
                                                                       eccentricity_threshold_muon=discrimination_criteria["electron_muon"].get("eccentricity_threshold_muon", 0.99),
                                                                       area_threshold_muon=discrimination_criteria["electron_muon"].get("area_threshold_muon", 25),
                                                                       eccentricity_threshold_alpha=discrimination_criteria["electron_muon"].get("eccentricity_threshold_alpha", 0.70),
                                                                       solidity_threshold_alpha=discrimination_criteria["electron_muon"].get("solidity_threshold_alpha", 1),
                                                                       area_threshold_alpha=discrimination_criteria["electron_muon"].get("area_threshold_alpha", 9))
    
    # Safe arithmetic on small integer matrices
    electrons_image = (image_tracks.astype(np.int16) - Muon_matrix.astype(np.int16) - Alpha_correction_matrix.astype(np.int16))
    electrons_image = np.clip(electrons_image, 0, 255).astype(np.uint8)

    image_alpha = (image_alpha.astype(np.int16) + Alpha_correction_matrix.astype(np.int16))
    image_alpha = np.clip(image_alpha, 0, 255).astype(np.uint8)


    N_alpha += N_alpha_corr

    N_gamma = event_counting_photon(image_gamma)


    results = {"Counts": {
                "alpha": N_alpha,
                "electrons": N_electrons,
                "muons": N_muons,
                "gamma": N_gamma
            },
            "Images": None if not return_images else {
                "original": image,
                "alpha": image_alpha,
                "electrons": electrons_image,
                "muons": Muon_matrix,
                "gamma": image_gamma
            }
            }

    if plot:

        image_couleur = slice_Tot(data.to_numpy(), t, d_time) # Image coloriée par le TOT pour visualisation
        plot_results(image, image_alpha, electrons_image, Muon_matrix, image_gamma, image_couleur, block = block, save=save)

    if save[0]:
        outdir = Path(save[2]) / Path(save[1])
        outdir.mkdir(parents=True, exist_ok=True)
        np.save(outdir / "image_originale.npy", image.astype(np.uint8), allow_pickle=False)
        np.save(outdir / "image_alpha.npy", image_alpha.astype(np.uint8), allow_pickle=False)
        np.save(outdir / "image_electrons.npy", electrons_image.astype(np.uint8), allow_pickle=False)
        np.save(outdir / "image_muons.npy", Muon_matrix.astype(np.uint8), allow_pickle=False)
        np.save(outdir / "image_gamma.npy", image_gamma.astype(np.uint8), allow_pickle=False)


    return results

def compteur_particles_optimized(file = "None", t_min= None,t_max=None, d_time = 150,
                    is_sliced = False, return_slice = False, return_images = False,return_data_t_max=False,
                    discrimination_criteria = {"electron_muon":{}},
                    progress_bar = False,progress_callback=None,
                    stop_requested=None,
                    single_window=False):
    """
    Comptage optimisé de particules.
    
    Args:
        single_window: Si True, traite [t_min, t_max] comme une seule fenêtre (pour visualisation).
                      Si False, découpe en plusieurs fenêtres de taille d_time (mode global).
    """

    data = file if not(type(file) == str) else read(file)
    print("Starting slicing...")

     # fallback safe callbacks if none provided
    if progress_callback is None:
        def progress_callback(progress, message=""):
            return None
    if stop_requested is None:
        def stop_requested():
            return False
    
    # Initialize data_t_max_value to None
    data_t_max_value = None
    
    if is_sliced:
        images = file
    elif single_window:
        # Mode visualisation: une seule fenêtre [t_min, t_max]
        progress_callback(0.1, "Slicing single window")
        window_start = t_min if t_min is not None else 0.0
        window_end = t_max if t_max is not None else data.iloc[:, 1].max()
        data_t_max_value = data.iloc[:, 1].max() if return_data_t_max else None
        
        # Créer une seule image pour la fenêtre
        image = slice(data.to_numpy(), window_start, window_end - window_start)
        images = [image]
        progress_callback(1.0, "Slicing complete")
    else:
        # Mode global: plusieurs fenêtres
        images_build = optimised_slice(data.to_numpy(), d_time,t_min=t_min,t_max=t_max,progress_bar=progress_bar,
                                 return_data_t_max=return_data_t_max,
                                 progress_callback=progress_callback, stop_requested=stop_requested,)
        slice_start_times = None
        if return_data_t_max:
            # Vérifier que c'est bien un tuple ((slices, times), data_t_max)
            if isinstance(images_build, tuple) and len(images_build) == 2:
                slices_and_times, data_t_max_value = images_build
                if isinstance(slices_and_times, tuple) and len(slices_and_times) == 2:
                    images, slice_start_times = slices_and_times
                else:
                    images = slices_and_times if isinstance(slices_and_times, list) else []
            else:
                images = images_build if isinstance(images_build, list) else []
                data_t_max_value = None
        else:
            # No return_data_t_max: expect (slices, times) tuple
            if isinstance(images_build, tuple) and len(images_build) == 2:
                images, slice_start_times = images_build
            else:
                images = images_build if isinstance(images_build, list) else []

    # number of windows (optimised_slice returns a list)
    n_windows = len(images)

    # accumulate totals in a dict for clarity and extensibility
    totals = {"alpha": 0, "electrons": 0, "muons": 0, "gamma": 0}
    
    # Store images from last window if requested
    last_images = None
    
    # Store slice start times for reuse in visualisation
    slice_times_for_return = slice_start_times

    print("\n","Starting optimized counting over", n_windows, "windows.\n")
    for i, image in enumerate(images):
        # check external stop request
        try:
            if callable(stop_requested) and stop_requested():
                try:
                    progress_callback(i / n_windows if n_windows else 1.0, "Counting stopped by request")
                except Exception:
                    pass
                break
        except Exception:
            pass

        if progress_bar:
            print(f"Processed window {i+1}/{n_windows}", end='\r', flush=True)
        results = compteur_particles(image, is_slice=True,
                    return_images=return_images,  # Passer le paramètre return_images
                    discrimination_criteria=discrimination_criteria)

        counts = results.get("Counts", {})
        result_images = results.get("Images", None)
        
        # Store images from the last (or only) window
        if return_images and result_images is not None:
            last_images = result_images
        
        
        for k in totals.keys():
            try:
                totals[k] += int(counts.get(k, 0) or 0)
            except Exception:
                # ignore malformed values and treat as zero
                pass
        # report progress for counting stage (fraction of windows done)
        try:
            progress_callback((i+1)/n_windows if n_windows else 1.0, f"Counting window")
        except Exception:
            pass

    # final progress
    try:
        progress_callback(1.0, "Counting complete")
    except Exception:
        pass
    # return a results-like dict for compatibility
    return {"Counts": totals, "Images": last_images if return_images else None, "Slices": images if return_slice else None, "SliceTimes": slice_times_for_return if return_slice else None, "data_t_max": data_t_max_value}
    




if __name__ == "__main__":
    # Lecture des données et création de l'image binaire
    file = "C:/Users/Félix/Desktop/Programmation/Projet_cea/Particle_tracking_algorithme_project_M1-2-IN/DATA-20251022T080148Z-1-001/DATA/alpha/60sec_alpha_39kbq_2.5cm_r0.t3pa"
    
    data = read(file)
    dt = 150
    n_windows = int(np.ceil(data.iloc[:, 1].max() / dt)) if dt > 0 else 1
    N_alpha_total = 0
    N_electrons_total = 0      
    N_muons_total = 0
    N_gamma_total = 0
    time_start = time.time()
    for i in range(n_windows):
        counts = compteur_particles(file, t=i*dt, d_time=dt, save=[False,"Test_alpha_qui_passe_pour_dafuk",Path.cwd()])
        #spliting results
        N_alpha, N_electrons, N_muons, N_gamma = counts["Counts"]["alpha"], counts["Counts"]["electrons"], counts["Counts"]["muons"], counts["Counts"]["gamma"]
        N_alpha_total += N_alpha
        N_electrons_total += N_electrons
        N_muons_total += N_muons
        N_gamma_total += N_gamma
        print(f"Fenêtre {i+1}/{n_windows} en {time.time() - time_start:.2f}s : Alpha={N_alpha}, Electrons={N_electrons}, Muons={N_muons}, Gamma={N_gamma}", end='\n ', flush=True)
    print("\nRésultats totaux sur toutes les fenêtres temporelles :")
    print(f"Temps total de traitement : {time.time() - time_start:.2f}s")
    print(f"Nombre de particules alpha détectées : {N_alpha_total}")
    print(f"Nombre de particules électrons détectées : {N_electrons_total}")
    print(f"Nombre de particules muons détectées : {N_muons_total}")
    print(f"Nombre de particules gamma détectées : {N_gamma_total}")