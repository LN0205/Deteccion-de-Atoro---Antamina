import cv2
import torch
import pandas
import numpy as np
import matplotlib.pyplot as plt
import time
import os
from findfilt import Filt
def apply_clahe(image):
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2Lab)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(1, 1))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_Lab2RGB)

start_time = time.time()
k=r"E:\repos\yolov8_training\Depth-Anything-V2\DATASETCHUTE\ERRORES\filtro_1-2024-10-25_13-49-51-atoro.png"
print(k) 
band2=False
examp=Filt(k)  
examp.label()
image_rgb = cv2.cvtColor(cv2.imread(k), cv2.COLOR_BGR2RGB)

new_width = 960  # New width in pixels
new_height = 540  # New height in pixels
new_size = (new_width, new_height)
resized_image = cv2.resize(image_rgb, new_size)
puntos=np.array([[712, 0], [690, 175], [601, 377], [582, 464], [612, 540], [960, 540], [960, 0]])
#cv2.fillPoly(resized_image, [puntos], color=(0, 0, 0))
enhanced_image = apply_clahe(resized_image)
gris_image = cv2.cvtColor(enhanced_image, cv2.COLOR_RGB2GRAY)
points=examp.corde()
examp.label()
umbl=examp.umbral()
#print(umbl)
imgleft = gris_image[points[0][0]:points[0][2], points[0][1]:points[0][3]]
imgright = gris_image[points[1][0]:points[1][2], points[1][1]:points[1][3]]
promedioleft = np.mean(imgleft, axis=(0, 1))
promedioright = np.mean(imgright, axis=(0, 1))
if (np.any(umbl[0] - promedioleft < 0)) and (np.any(umbl[1] - promedioright < 0)):
    print("Hay atoro en el lado derecho e izquierdo de la imagen: ", k)
    band1=True
    #cont1=cont1+1
elif (np.any(umbl[0] - promedioleft < 0)) and (np.any(umbl[1] - promedioright >= 0)):
    print("Hay atoro en el lado izquierdo de la imagen:  ", k)
    band1=True
    #cont1=cont1+1
elif (np.any(umbl[0] - promedioleft >= 0)) and (np.any(umbl[1] - promedioright < 0)):
    print("Hay atoro en el lado derecho de la imagen:  ", k)
    band1=True
    #cont1=cont1+1
else:
    #print("No hay atoro en la imagen:  ", k)
    band1=False

end_time = time.time()

# Calcula el tiempo de ejecución
execution_time = end_time - start_time
print("Tiempo de ejecucion :", execution_time)
plt.imshow(gris_image)
plt.axis('off')  # Hide axes
plt.title("Displayed Image")  # Optional: add a title
plt.show()

