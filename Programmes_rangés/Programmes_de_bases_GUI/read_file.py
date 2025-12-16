# Contient l'ensemble des fonctions nécessaires à l'acquisition des fichiers de données
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def read(file):
    """Read a whitespace-separated data file and return the relevant columns.

    Parameters
    ----------
    file : str
        Path to the input file to read.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing only columns 1, 2 and 3 (0-based indexing).
    """
    df = pd.read_csv(file, sep=r"\s+", header=0)
    return df.iloc[:, [1,2,3]]


def slice(data,t,d_time):
    """Make a binary 256x256 image from detections within a time window.

    Parameters
    ----------
    data : ndarray
        NxM array with detection rows. Column 0 is the pixel index, column 1 is time.
    t : float
        Start time of the window.
    d_time : float
        Duration of the time window.

    Returns
    -------
    numpy.ndarray
        256x256 binary image where pixels with at least one detection in the time
        window are set to 1.
    """
    image = np.zeros((256,256)) #Création image vide 

    mask = (data[:, 1] >= t) & (data[:, 1] <= t + d_time) # Filtrage des données selon le temps
    data_cut = data[mask]

      # Vectorised assignment: compute coordinates for all detections then assign
    # This avoids a Python loop and is significantly faster for many detections.
    if data_cut.size == 0:
        return image  # return empty image if no data in time window
    indices = data_cut[:, 0].astype(np.int64)
    # clip indices to valid range in case of malformed input
    indices = np.clip(indices, 0, 256 * 256 - 1)
    xs = indices // 256
    ys = indices % 256
    values = np.ones_like(data_cut[:, 2])
    # advanced (fancy) indexing: later entries in values will overwrite earlier
    # ones at the same (x,y), which matches the behaviour of the explicit loop
    image[xs, ys] = values
    return image

def slice_Tot(data,t,d_time):
    """Make a 256x256 image where pixel values are set to the 'TOT' value.

    This function uses column 2 of the input data as the pixel intensity (TOT).

    Parameters
    ----------
    data : ndarray
        NxM array with detection rows. Column 0 is the pixel index, column 1 is time,
        column 2 is the TOT (intensity) value.
    time : float
        Start time of the window.
    d_time : float
        Duration of the time window.

    Returns
    -------
    numpy.ndarray
        256x256 image where each pixel contains the TOT value of the detection in
        the time window (last assignment wins if multiple detections land on same pixel).
    """
    image = np.zeros((256,256)) #Création image vide 

    mask = (data[:, 1] >= t) & (data[:, 1] < t + d_time) # Filtrage des données selon le temps
    data_cut = data[mask]

    # Vectorised assignment: compute coordinates for all detections then assign
    # This avoids a Python loop and is significantly faster for many detections.
    if data_cut.size == 0:
        return image  # return empty image if no data in time window
    indices = data_cut[:, 0].astype(np.int64)
    # clip indices to valid range in case of malformed input
    indices = np.clip(indices, 0, 256 * 256 - 1)
    xs = indices // 256
    ys = indices % 256
    values = data_cut[:, 2]
    # advanced (fancy) indexing: later entries in values will overwrite earlier
    # ones at the same (x,y), which matches the behaviour of the explicit loop
    image[xs, ys] = values
    return image


def optimised_slice(data, dt, TOT=False, t_min=None, t_max=None,
                    progress_bar=False, progress_callback=None,
                    return_data_t_max=False,
                    stop_requested=None):
     
    """Slice the input detections into multiple time-window images.

    This function examines the timestamps present in ``data`` and builds a list
    of 256x256 images (binary or TOT-valued depending on ``TOT``) where each
    image corresponds to a time window of duration ``dt`` starting at selected
    time points. The intent is to avoid producing overlapping windows for
    time points that are closer than ``dt``.

    Behavior summary
    - If ``TOT`` is False (default) the helper ``slice`` is used to produce
      binary images (0/1). If ``TOT`` is True the helper ``slice_Tot`` is
      used to place TOT values into the images.
    - The function first determines the unique timestamp values within the
      optional interval [``t_min``, ``t_max``] (or over the whole dataset if
      they are not provided).
    - It then iterates through the sorted unique timestamps and creates a
      window image at the first timestamp and thereafter whenever the next
      timestamp is more than ``dt`` after the current one. This ensures
      successive windows start at timestamp locations spaced by at least ``dt``.

    Parameters
    ----------
    data : ndarray
        NxM array with detection rows. Column 0 is the pixel index, column 1 is time
        and column 2 (if present) may be TOT/intensity used by ``slice_Tot``.
    dt : float
        Duration of each time window.
    TOT : bool, optional
        If True use ``slice_Tot`` (store TOT values), otherwise use the
        binary ``slice`` function. Default False.
    t_min, t_max : float, optional
        If provided, restrict the analyzed timestamps to the interval
        [t_min, t_max]. By default the full min/max from data[:,1] is used.

    Returns
    -------
    list of ndarray
        List of 256x256 images (dtype float) corresponding to selected windows.

    Notes / edge cases
    -------------------
    - If no timestamps fall inside [t_min, t_max] an empty list is returned.
    - This function relies on the helper functions ``slice`` and ``slice_Tot``
      defined in this module.
    - The function uses unique timestamps as potential window starts; it does
      not generate a dense tiling of windows.
    """
     # make sure we have safe callables
    if progress_callback is None:
        def progress_callback(progress, message=""):
            return None
    if stop_requested is None:
        def stop_requested():
            return False

      

    # determine time interval to consider
    start_time = t_min if t_min is not None else data[:, 1].min()
    end_time = t_max if t_max is not None else data[:, 1].max()

    # restrict to points inside interval and extract arrays we'll reuse
    in_interval = (data[:, 1] >= start_time) & (data[:, 1] <= end_time)
    if not np.any(in_interval):
        return []

    times = data[in_interval, 1]
    indices_all = data[in_interval, 0].astype(np.int64)
    # values (TOT) may be present in column 2; keep reference even for binary mode
    values_all = data[in_interval, 2]

    # unique sorted candidate start times
    cutting_points = np.unique(times)
    if cutting_points.size == 0:
        return []

    # choose starting timestamps spaced by at least dt (vectorized selection)
    # always include the first unique timestamp
    gaps = np.diff(cutting_points)
    starts_idx = np.concatenate(([0], np.nonzero(gaps > dt)[0] + 1))
    start_times = cutting_points[starts_idx]

    Npix = 256 * 256 
    slices = []

    total = len(start_times)
    # initial progress
    try:
        progress_callback(0.0, f"Preparing {total} windows")
    except Exception:
        pass
    # For each chosen start time, build the image by selecting the detections
    # that fall inside [t0, t0+dt) (TOT mode) or [t0, t0+dt] (binary mode) and
    # assign values vectorized (avoids Python-level loops and repeated work).
    for i, t0 in enumerate(start_times):
        # allow external stop request
        try:
            if callable(stop_requested) and stop_requested():
                try:
                    progress_callback(i/total if total else 1.0, "Stopped by request")
                except Exception:
                    pass
                break
        except Exception:
            pass

        if progress_bar:
            print(f"Building window {i+1}/{len(start_times)}", end='\r', flush=True)

        if TOT:
            mask_win = (times >= t0) & (times < (t0 + dt))
        else:
            mask_win = (times >= t0) & (times < (t0 + dt))

        if not np.any(mask_win):
            # empty window -> zero image
            slices.append(np.zeros((256, 256)))
            continue

        idxs = indices_all[mask_win]
        # clip indices to valid domain
        idxs = np.clip(idxs, 0, Npix-1)
        flat = idxs.astype(np.int64)

        if TOT:
            vals = values_all[mask_win]
            # create flat image and assign values (last-write wins w.r.t. data order)
            flat_img = np.zeros(Npix, dtype=vals.dtype)
            flat_img[flat] = vals
        else:
            # binary image: a hit sets pixel to 1
            flat_img = np.zeros(Npix, dtype=np.uint8)
            flat_img[flat] = 1
        slices.append(flat_img.reshape((256, 256)))

        try:
            progress_callback((i+1)/total, f"Building window")
        except Exception:
            pass
    if progress_bar:
        print('\n')
    try:
        progress_callback(1.0, "Slicing complete")
    except Exception:
        pass
    # Build return tuple with start_times included
    result = (slices, start_times) if len(slices) > 0 else (slices, np.array([]))
    if return_data_t_max:
        return result, data[:, 1].max()
    return result