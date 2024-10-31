import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from findfilt import Filt
from Probando1023 import detectvoid
from findfilt import apply_clahe

#k=r"E:\repos\yolov8_training\Depth-Anything-V2\DATASETCHUTE\ERRORES\filtro_1-2024-10-24-10h24m35s711.png "
#k=r"E:\repos\yolov8_training\Depth-Anything-V2\DATASETCHUTE\ERRORES\filtro_1-2024-10-25_13-47-16.png"
#k=r"E:\repos\yolov8_training\Depth-Anything-V2\DATASETCHUTE\ERRORES\filtro_1-2024-10-25_13-14-23-atoro.png"
#k=r"E:\repos\yolov8_training\Depth-Anything-V2\DATASETCHUTE\ERRORES\filtro_1-2024-10-25_13-39-16.png"
#k=r"E:\repos\yolov8_training\Depth-Anything-V2\DATASETCHUTE\ERRORES\filtro_1-2024-10-25_12-56-03-atoro.png"
#k=r"E:\repos\yolov8_training\Depth-Anything-V2\DATASETCHUTE\ERRORES\filtro_1-2024-10-25_13-10-11-atoro (1).png"
#k=r"E:\repos\yolov8_training\Depth-Anything-V2\DATASETCHUTE\ERRORES\filtro_1-2024-10-25_13-49-51-atoro.png"
#k=r"E:\repos\yolov8_training\Depth-Anything-V2\DATASETCHUTE\F1\filtro_1-2024-10-22_13-48-36.png"
#k=r"E:\repos\yolov8_training\Depth-Anything-V2\DATASETCHUTE\F1\filtro_1-2024-10-25_13-47-16.png"
#k=r"E:\repos\yolov8_training\Depth-Anything-V2\DATASETCHUTE\F1\filtro_1-2024-10-21_01-17-38.png"
#k=r"E:\repos\yolov8_training\DATASETSDN\antamina-filter-segmentation.v4i.yolov8\train\images\filtro_3-2024-01-30_04-42-47_png.rf.01689c28e5c3b9c8432440149730aa18.jpg"
#---------------------------------------------------------------------------------------------------------------------
#k=r"E:\repos\yolov8_training\Depth-Anything-V2\DATASETCHUTE\F1\filtro_1-2024-09-01_11-35-47.png"
#k=r"E:\repos\yolov8_training\Depth-Anything-V2\DATASETCHUTE\F1\filtro_1-2024-08-03_18-42-37.png"
#k=r"E:\repos\yolov8_training\Depth-Anything-V2\DATASETCHUTE\F1\filtro_1-2024-04-11_10-54-32.png"
#k=r"E:\repos\yolov8_training\Depth-Anything-V2\DATASETCHUTE\F1\filtro_1-2024-04-12_15-42-03.png"
#k=r"E:\repos\yolov8_training\Depth-Anything-V2\DATASETCHUTE\F1\filtro_1-2024-03-27_23-34-15.png"
#k=r"E:\repos\yolov8_training\DATASETSDN\antamina-filter-segmentation.v4i.yolov8\train\images\filtro_4-2024-01-19_18-52-08-atoro_png.rf.3bdf09ad87cc49a44d4f693dfe1a4ce3.jpg"
#k=r"E:\repos\yolov8_training\DATASETSDN\antamina-filter-segmentation.v4i.yolov8\train\images\filtro_4-2024-01-19_18-52-08-atoro_png.rf.990b057e21f7faf6331e32e818177917.jpg"
#k=r"E:\repos\yolov8_training\Depth-Anything-V2\DATASETCHUTE\ERRORES\filtro_1-2024-10-25_13-47-16.png"
#k=r"E:\repos\yolov8_training\Depth-Anything-V2\DATASETCHUTE\ERRORES\filtro_1-2024-10-24-10h24m35s711.png"
#k=r"E:\repos\yolov8_training\Depth-Anything-V2\DATASETCHUTE\F1\filtro_1-2024-08-31_21-23-03.png"
#k=r"E:\repos\yolov8_training\Depth-Anything-V2\DATASETCHUTE\F1\filtro_1-2024-08-10_13-02-55.png"

#Filtro2
#k=r"E:\repos\yolov8_training\DATASETSDN\antamina-filter-segmentation.v4i.yolov8\train\images\filtro_2-2024-01-18_19-09-43_png.rf.5f9cbbb378c506e4377e0baf3e819124.jpg"

#Filtro3
k=r"E:\repos\yolov8_training\DATASETSDN\antamina-filter-segmentation.v4i.yolov8\train\images\filtro_3-2024-01-21_05-24-22_png.rf.1567e4ccec2cd86cc99801e1fbe8d128.jpg"
examp=Filt(k)  
label=examp.label()
cordl,cordr=examp.corde()
mask1,mask2,mask3=examp.mask()
image_rgb = cv2.cvtColor(cv2.imread(k), cv2.COLOR_BGR2RGB)
new_width = 960  # New width in pixels
new_height = 540  # New height in pixels
new_size = (new_width, new_height)
resized_image = cv2.resize(image_rgb, new_size)
maskl=np.zeros(resized_image.shape, dtype=np.uint8)
cv2.fillPoly(maskl, [cordl], (255,255,255))
bool_maskl = maskl[:, :, 0] == 255
maskr=np.zeros(resized_image.shape, dtype=np.uint8)
cv2.fillPoly(maskr, [cordr], (255,255,255))
bool_maskr = maskr[:, :, 0] == 255
maskcnt1=np.zeros(resized_image.shape, dtype=np.uint8)
cv2.fillPoly(maskcnt1, [mask1], (255,255,255))
bool_maskcnt1 = maskcnt1[:, :, 0] == 255
#................
maskcnt2=np.zeros(resized_image.shape, dtype=np.uint8)
cv2.fillPoly(maskcnt2, [mask2], (255,255,255))    
bool_maskcnt2 = maskcnt2[:, :, 0] == 255
#..................................
maskcnt3=np.zeros(resized_image.shape, dtype=np.uint8)
cv2.fillPoly(maskcnt3, [mask3], (255,255,255))
bool_maskcnt3 = maskcnt3[:, :, 0] == 255
enhanced_image = apply_clahe(resized_image)
lum1 = enhanced_image[bool_maskcnt1, 0]
average1= np.mean(lum1)
lum2 = enhanced_image[bool_maskcnt2, 0]
average2= np.mean(lum2)
adj=examp.adjust()
adjustment1 = ((average1 - adj[0])/adj[1]).astype(np.float32)
#................................................
adjustment2=((average2 - adj[2])/adj[3]).astype(np.float32)
#................................................

enhanced_image[bool_maskr, 0] = np.clip(enhanced_image[bool_maskr, 0] - adjustment1,0,255)
enhanced_image[bool_maskcnt3, 0] = np.clip(enhanced_image[bool_maskcnt3, 0] - adjustment1,0,255)
#...........................
enhanced_image[bool_maskl, 0] = np.clip(enhanced_image[bool_maskl, 0] - adjustment2,0,255)
#.........................
enhanced_image[:, :, 0] = np.clip(enhanced_image[:, :, 0], 0, 255)
#------------------------------------------------------------
rgb_image = cv2.cvtColor(enhanced_image, cv2.COLOR_Lab2RGB)
gris_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
#------------------------------------------------------------
#gris_image = cv2.cvtColor(enhanced_image, cv2.COLOR_RGB2GRAY)
points=examp.corde()
examp.label()
umbl=examp.umbral()
promedioleft= np.mean(gris_image[bool_maskl])
promedioright= np.mean(gris_image[bool_maskr])
promediopat=np.mean(gris_image[bool_maskcnt3])
if (promedioleft-promediopat > umbl[0]) and (promedioright-promediopat > umbl[1]):
        print("Hay atoro en el lado derecho e izquierdo de la imagen: ", k)
        bandl=True
        bandr=True
        #cont1=cont1+1
elif (promedioleft-promediopat > umbl[0]) and (promedioright-promediopat <= umbl[1]):
        print("Hay atoro en el lado  izquierdo de la imagen: ", k)
        bandl=True
        bandr=False
        #cont1=cont1+1
elif (promedioleft-promediopat <= umbl[0]) and (promedioright-promediopat > umbl[1]):
        print("Hay atoro en el lado derecho: ", k)
        bandl=False
        bandr=True
        #cont1=cont1+1
else:
        #print("No hay atoro en la imagen:  ", k)
        bandl=False
        bandr=False

print(average1)      
print(average2)   
print(adjustment1)
print(adjustment2)

print("promdedio left ", promedioleft)
print("promdedio patt ", promediopat)
print("promdedio right ",promedioright )

plt.imshow(gris_image)
plt.axis('off')  # Opcional: Ocultar los ejes
plt.show()