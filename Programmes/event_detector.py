# -*- coding: utf-8 -*-
"""
Created on Wed Oct 22 09:48:09 2025

@author: sebwi
"""

import numpy as np 
import matplotlib.pyplot as plt
from scipy.ndimage import label

#%%
df = np.load('image_alpha_2.npy')

plt.figure(figsize=(15,15))
plt.imshow(df, cmap='gray', origin='upper')  # origin='upper' pour que (0,0) soit en haut à gauche
plt.title("Matrice image")
plt.xlabel("y")
plt.ylabel("x")
plt.colorbar(label='Valeur')
plt.show()


#%% 
def clustering(image):

    rows = len(df)  # nb de ligne
    cols = len(df[0]) if rows > 0 else 0    # nb de colone

    visited = [[False for _ in range(cols)] for _ in range(rows)]   # matrice de false qui devient true quand cluster
    directions = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]     # position des pixels qu'on check autour de celui sur lequel on est placé
    def dfs(r, c):  
        stack = [(r, c)]
        while stack:
            x, y = stack.pop()
            if visited[x][y]:
                continue
            visited[x][y] = True
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if 0 <= nx < rows and 0 <= ny < cols:
                    if image[nx][ny] == 1 and not visited[nx][ny]:
                        stack.append((nx, ny))
    cluster_count = 0
    for i in range(rows):
        for j in range(cols):
            if image[i][j] == 1 and not visited[i][j]:
                dfs(i, j)
                cluster_count += 1
    return cluster_count    



print(clustering(df))

# Alternative + simple 
clusters, n_clusters = label(df)
print(n_clusters)

















