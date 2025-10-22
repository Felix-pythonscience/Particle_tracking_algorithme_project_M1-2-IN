# Contient l'ensemble des fonctions nécessaires à l'acquisition des fichiers de données


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def read(file):
    df = pd.read_csv(file, sep=r"\s+", header=0)
    return df.iloc[:, [1,2,3]]


def slice(data,time,d_time):
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
    image = np.zeros((256,256)) #Création image vide 

    mask = (data[:, 1] >= time) & (data[:, 1] <= time + d_time) # Filtrage des données selon le temps
    data_cut = data[mask]

    for detection in data_cut :  # Boucle pour mettre les pixels à 1
        index = int(detection[0]) #Laisser -1 pour avoir un index correct si il commence à 1
        x = index // 256
        y = index % 256
        image[x,y] = detection[2]
    return image