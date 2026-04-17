# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 15:14:47 2026

@author: alexa
"""


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image


#mesures à distance quasi nulle, à 1000V
# (0,0) = en bas à gauche
# (4, 0) = en bas à droite 
# flèche vers le bas

res_matrix = np.array([
    [24822, None, 26618, None, 26111], 
    [None, 26545, None, 20955, None],
    [22778, None, 24192, None, 24427],
    [None, 25611, None, 16523, None],   #15868
    [22944, None, 18206, None, 20377]
    ], dtype = object)





# Conversion des 'None' en 'NaN' et conversion en type float pour le calcul
z_data = np.array(res_matrix, dtype=float)
z_data = z_data / 120
#np.round(z_data)
#z_data = np.array(z_data, dtype=int)

# Création des coordonnées X et Y (index de la matrice)
rows, cols = z_data.shape
x, y = np.meshgrid(np.arange(cols), np.arange(rows))

# Configuration du graphique 3D
plt.close('all')
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Affichage des points (Scatter plot) ou des barres (Bar3d)
# On filtre les NaN pour éviter les warnings lors du rendu
mask = ~np.isnan(z_data)
sc = ax.scatter(x[mask], y[mask], z_data[mask], c=z_data[mask], cmap='viridis', s=100)

# Habillage du graphique
ax.set_title("Visualisation 3D des mesures sur la feuille")
ax.set_xlabel("Colonnes (X)")
ax.set_ylabel("Lignes (Y)")
ax.set_zlabel("Valeurs (Z)")

# Ajout d'une barre de couleur pour l'échelle
plt.colorbar(sc, ax=ax, shrink=0.5, aspect=5)

plt.show()



# =============================================================================
# Export en .bmp
# =============================================================================

# Création d'une matrice 640x480 remplie de zéros (fond noir)
width, height = 640, 480
img_array = np.zeros((height, width), dtype=np.uint8)

# Remplissage des zones (Slicing)
# Note : En NumPy, on accède par [y_start:y_end, x_start:x_end]

img_array[103:165, 148:225] = int(np.round(24822 / 120))
img_array[103:165, 316:393] = int(np.round(26618 / 120))
img_array[103:165, 486:563] = int(np.round(26111 / 120))

img_array[266:328, 148:225] = int(np.round(22778 / 120))
img_array[266:328, 316:393] = int(np.round(24192 / 120))
img_array[266:328, 486:563] = int(np.round(24427 / 120))

img_array[418:480, 148:225] = int(np.round(22944 / 120))
img_array[418:480, 316:393] = int(np.round(18206 / 120))
img_array[418:480, 486:563] = int(np.round(20377 / 120))

img_array[192:242, 241:303] = int(np.round(26545 / 120))
img_array[192:242, 410:476] = int(np.round(20955 / 120))

img_array[355:407, 241:305] = int(np.round(25611 / 120))
img_array[305:407, 410:476] = int(np.round(16523 / 120))

# Conversion et sauvegarde 
# ATTENTION : Le format BMP standard est souvent en 8-bit (0-255) ou 24-bit RGB.
# Tes valeurs (24000+) dépassent la limite d'une image classique.
# Pour conserver les données brutes, on peut sauvegarder en mode "I" (32-bit signed integer)
final_img = Image.fromarray(img_array, mode='L')
final_img.save("feuille_mesures.bmp")

print("Image BMP générée avec succès !")




# A = [12/330, 12/335, 11.9/335]

# np.mean(A)
# Out[44]: 0.035902306648575306

# np.std(A)
# Out[45]: 0.0003482293361360478