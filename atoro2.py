import cv2
import torch
import pandas
import numpy as np
import matplotlib.pyplot as plt
import time
import os
from findfilt import Filt
from Probando1023 import detectvoid

#carp_imag=r"E:\repos\yolov8_training\Depth-Anything-V2\DATASETCHUTE\F1"
#carp_imag=r"E:\repos\yolov8_training\Depth-Anything-V2\DATASETCHUTE\F2"
carp_imag=r"E:\repos\yolov8_training\DATASETSDN\antamina-filter-segmentation.v4i.yolov8\train\images"
#carp_imag=r"E:\repos\yolov8_training\Depth-Anything-V2\DATASETCHUTE\F1"
#carp_lab=r"C:\repos\antamina-filter-clogging-preventer\datasets\antamina-filter-segmentation.v4i.yolov8\valid\labels"
#carp_imag=r"E:\repos\yolov8_training\Depth-Anything-V2\DATASETCHUTE\F1"
paths = []
cont1=0
for archivo in os.listdir(carp_imag):
    if archivo.endswith(".jpg") or archivo.endswith(".png"):
        paths.append(os.path.join(carp_imag, archivo))
for k in paths:
    label,bandl,bandr,promedioleft,promediopat,promedioright,iluaverage1,iluaverage2=detectvoid(k)
    if (bandl)  and (bandr):
        print("Hay atoro en el lado derecho e izquierdo de la imagen: ", k)
        cont1=cont1+1
    elif (bandl)  and (not bandr) :
        print("Hay atoro en el lado izquierdo de la imagen:  ", k)
        cont1=cont1+1
    elif ((not bandl)  and (bandr)):
        print("Hay atoro en el lado derecho de la imagen:  ", k)
        cont1=cont1+1
    else:
        #print("No hay atoro en la imagen:  ", k)
        band1=False
print("Numeror de atoros detectados umbralizacion :",cont1)
print("Numeror de total de imagenes :",len(paths))
