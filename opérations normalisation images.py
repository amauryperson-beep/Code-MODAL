# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 17:31:13 2026

@author: alexa
"""

from PIL import Image
import numpy as np

# Charger ton image générée précédemment (si elle n'est plus en mémoire)
img1 = np.array(Image.open("feuille_mesures.bmp").convert('L'), dtype=np.float32)

# Charger ton image importée
img_importee = Image.open("pixels.bmp").convert('L')

img2 = np.array(img_importee, dtype=np.float32)




L = []
for l in img1:
    for i in l:
        if i!=0:
            L.append(i)
L = np.array(L)
L_2 = []
for l in img2:
    for i in l:
        if i!=0:
            L_2.append(i)
L_2 = np.array(L_2)


norm = max(L_2)/min(L)
print(norm)

# Chaque pixel de 'result' est égal à la somme des pixels correspondants
result = np.where(img1*img2>0,(img2 / img1)*255/norm,0)

# On borne entre 0 et 255 et on convertit en entiers 8-bits
result_final = np.clip(result, 0, 255).astype(np.uint8)

# Sauvegarde
Image.fromarray(result_final).save("sources_normalisees.bmp")