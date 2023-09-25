import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

'''
GET_DATA_FROM_TXT_FILE
Arguments :
    filename - un chemin (str) vers l'emplacement des données
Retourne :
    points - une matrice de points où chaque ligne est soit :
        a) les coordonnées homogènes (x,y,1) si les données sont en 2D
        b) les coordonnées (x,y,z) si les données sont en 3D
    use_subset - utilise un sous-ensemble prédéfini (codé en dur pour l'instant)
'''
def get_data_from_txt_file(filename, use_subset = False):
    with open(filename) as f:
            lines = f.read().splitlines()
    number_pts = int(lines[0])

    points = np.ones((number_pts, 3))
    for i in range(number_pts):
        split_arr = lines[i+1].split()
        if len(split_arr) == 2:
            y, x = split_arr
        else:
            x, y, z = split_arr
            points[i,2] = z
        points[i,0] = x 
        points[i,1] = y
    return points

'''
CALCULER_IMAGE_RECTIFIÉE
Arguments :
    im - une image
    H - une matrice d'homographie qui rectifie l'image
Retourne :
    new_image - une nouvelle matrice d'image après application de l'homographie
    offset - le décalage dans l'image.
'''
def compute_rectified_image(im, H):
    new_x = np.zeros(im.shape[:2])
    new_y = np.zeros(im.shape[:2])
    for y in range(im.shape[0]): # height
        for x in range(im.shape[1]): # width
            new_location = H.dot([x, y, 1])
            new_location /= new_location[2]
            new_x[y,x] = new_location[0]
            new_y[y,x] = new_location[1]
    offsets = (new_x.min(), new_y.min())
    new_x -= offsets[0]
    new_y -= offsets[1]
    new_dims = (int(np.ceil(new_y.max()))+1,int(np.ceil(new_x.max()))+1)

    H_inv = np.linalg.inv(H)
    new_image = np.zeros(new_dims)

    for y in range(new_dims[0]):
        for x in range(new_dims[1]):
            old_location = H_inv.dot([x+offsets[0], y+offsets[1], 1])
            old_location /= old_location[2]
            old_x = int(old_location[0])
            old_y = int(old_location[1])
            if old_x >= 0 and old_x < im.shape[1] and old_y >= 0 and old_y < im.shape[0]:
                new_image[y,x] = im[old_y, old_x]

    return new_image, offsets

'''
SCATTER_3D_AXIS_EQUAL
Arguments :
    X - les coordonnées sur l'axe des x (N long vector)
    Y - les coordonnées sur l'axe des y (N long vector)
    Z - les coordonnées sur l'axe des z (N long vector)
    ax - l'axe de pyplot
Retourne :
    Rien ; à la place, trace les points de (X, Y, Z) tels que les axes sont égaux.
'''
def scatter_3D_axis_equal(X, Y, Z, ax):
    ax.scatter(X, Y, Z)
    max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max() / 2.0

    mid_x = (X.max()+X.min()) * 0.5
    mid_y = (Y.max()+Y.min()) * 0.5
    mid_z = (Z.max()+Z.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
