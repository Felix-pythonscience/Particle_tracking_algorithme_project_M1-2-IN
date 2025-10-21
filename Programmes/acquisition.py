import pandas as pd
import numpy as np


file = ""
dt =


df = pd.read_csv(file, sep=r"\s+", header=0)
columns = df.iloc[:, [1, 2]]
index= columns[0]
time=  columns[1]

 
image = np.zeros((256,256))
for detection in detections :
    x = index//256
    y = time%256
    image[x,y] = 1


