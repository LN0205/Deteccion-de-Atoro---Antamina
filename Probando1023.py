import cv2
import torch
import pandas
import numpy as np
import matplotlib.pyplot as plt
import time
import os
from findfilt import apply_clahe
from findfilt import Filt
def detectvoid(k):
    #Defino las configuraciones
    examp=Filt(k)  
    label=examp.label()
    cordl,cordr=examp.corde()
    mask1,mask2,mask3=examp.mask()
    image_rgb = cv2.cvtColor(cv2.imread(k), cv2.COLOR_BGR2RGB)
    new_width = 960
    new_height = 540 
    new_size = (new_width, new_height)
    resized_image = cv2.resize(image_rgb, new_size)
    #Creo las mascaras
    maskl=np.zeros(resized_image.shape, dtype=np.uint8)
    cv2.fillPoly(maskl, [cordl], (255,255,255))
    bool_maskl = maskl[:, :, 0] == 255
    maskr=np.zeros(resized_image.shape, dtype=np.uint8)
    cv2.fillPoly(maskr, [cordr], (255,255,255))
    bool_maskr = maskr[:, :, 0] == 255
    maskcnt1=np.zeros(resized_image.shape, dtype=np.uint8)
    cv2.fillPoly(maskcnt1, [mask1], (255,255,255))
    bool_maskcnt1 = maskcnt1[:, :, 0] == 255
    maskcnt2=np.zeros(resized_image.shape, dtype=np.uint8)
    cv2.fillPoly(maskcnt2, [mask2], (255,255,255))
    bool_maskcnt2 = maskcnt2[:, :, 0] == 255
    maskcnt3=np.zeros(resized_image.shape, dtype=np.uint8)
    cv2.fillPoly(maskcnt3, [mask3], (255,255,255))
    bool_maskcnt3 = maskcnt3[:, :, 0] == 255
    #Aplico los filtros
    enhanced_image = apply_clahe(resized_image)
    lum1 = enhanced_image[bool_maskcnt1, 0]
    ilumaverage1= np.mean(lum1)
    lum2 = enhanced_image[bool_maskcnt2, 0]
    average2= np.mean(lum2)
    adj=examp.adjust()
    #Compenso la iluminacion
    adjustment1 = ((ilumaverage1 - adj[0])/adj[1]).astype(np.float32)
    adjustment2=((average2 - adj[2])/adj[3]).astype(np.float32)
    enhanced_image[bool_maskr, 0] = np.clip(enhanced_image[bool_maskr, 0] - adjustment1,0,255)
    enhanced_image[bool_maskcnt3, 0] = np.clip(enhanced_image[bool_maskcnt3, 0] - adjustment1,0,255)
    enhanced_image[bool_maskl, 0] = np.clip(enhanced_image[bool_maskl, 0] - adjustment2,0,255)
    enhanced_image[:, :, 0] = np.clip(enhanced_image[:, :, 0], 0, 255)
    #------------------------------------------------------------
    rgb_image = cv2.cvtColor(enhanced_image, cv2.COLOR_Lab2RGB)
    gris_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
    #------------------------------------------------------------
    #points=examp.corde()
    umbl=examp.umbral()
    examp.label()
    #Comparo los promedios
    promedioleft= np.mean(gris_image[bool_maskl])
    promedioright= np.mean(gris_image[bool_maskr])
    promediopat=np.mean(gris_image[bool_maskcnt3])
    if (promedioleft-promediopat > umbl[0]) and (promedioright-promediopat > umbl[1]):
        bandl=True
        bandr=True
        #cont1=cont1+1
    elif (promedioleft-promediopat > umbl[0]) and (promedioright-promediopat <= umbl[1]):
        bandl=True
        bandr=False
        #cont1=cont1+1
    elif (promedioleft-promediopat <= umbl[0]) and (promedioright-promediopat > umbl[1]):
        bandl=False
        bandr=True
        #cont1=cont1+1
    else:
        #print("No hay atoro en la imagen:  ", k)
        bandl=False
        bandr=False
    return label,bandl,bandr,promedioleft,promediopat,promedioright,ilumaverage1,average2



