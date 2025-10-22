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


def slice(data,time,d_time):
    """Make a binary 256x256 image from detections within a time window.

    Parameters
    ----------
    data : ndarray
        NxM array with detection rows. Column 0 is the pixel index, column 1 is time.
    time : float
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

    mask = (data[:, 1] >= time) & (data[:, 1] <= time + d_time) # Filtrage des données selon le temps
    data_cut = data[mask]

    for detection in data_cut :  # Boucle pour mettre les pixels à 1
        index = int(detection[0]) #Laisser -1 pour avoir un index correct si il commence à 1
        x = index // 256
        y = index % 256
        image[x,y] = 1
    return image

def slice_Tot(data,time,d_time):
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

    mask = (data[:, 1] >= time) & (data[:, 1] <= time + d_time) # Filtrage des données selon le temps
    data_cut = data[mask]

    for detection in data_cut :  # Boucle pour mettre les pixels à 1
        index = int(detection[0]) #Laisser -1 pour avoir un index correct si il commence à 1
        x = index // 256
        y = index % 256
        image[x,y] = detection[2]
    return image