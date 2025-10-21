import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def read(file):
    df = pd.read_csv(file, sep=r"\s+", header=0)
    return df.iloc[:, [1, 2]]


def slice(data,time,d_time):
    image = np.zeros((256,256)) #Création image vide 

    mask = (data[:, 1] >= time) & (data[:, 1] <= time + d_time) # Filtrage des données selon le temps
    data_cut = data[mask]

    for detection in data_cut :  # Boucle pour mettre les pixels à 1
        index = int(detection[0]) -1 #Laisser -1 pour avoir un index correct si il commence à 1
        x = index // 256
        y = index % 256
        image[x,y] = 1
    return image



# Test pour des gammas d'Am241
file = "C:/Users/stev2/Desktop/MINIPIX Bordeaux/60sec_alpha_39kbq_2.5cm_r0.t3pa"
data = read(file)
d_time = max(data.iloc[:,1])/500  # Diviser le temps

image = slice(data.to_numpy(),0, d_time)

# Affichage de la matrice comme image
plt.figure(figsize=(6,6))
plt.imshow(image, cmap='gray', origin='upper')  # origin='upper' pour que (0,0) soit en haut à gauche
plt.title("Matrice image")
plt.xlabel("y")
plt.ylabel("x")
plt.colorbar(label='Valeur')
plt.show()


