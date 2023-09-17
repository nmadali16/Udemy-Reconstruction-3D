import numpy as np

import matplotlib.pyplot as plt
import scipy.io as sio
from epipolar_utils import *

'''
LLS_EIGHT_POINT_ALG calcule la matrice fondamentale à partir des points correspondants en utilisant l'algorithme des 
l'algorithme des moindres carrés linéaires en huit points
Arguments :
    points1 - N points de la première image qui correspondent aux points2
    points2 - N points de la deuxième image qui correspondent aux points1

    Les points 1 et 2 proviennent de la méthode get_data_from_txt_file()
Retourne :
    F - la matrice fondamentale telle que (points2)^T * F * points1 = 0

'''
def lls_eight_point_alg(points1, points2):
    points_num = points2.shape[0]

    W = np.zeros((points_num, 9))
    for i in range(points_num):
        u1 = points1[i][0]
        v1 = points1[i][1]
        u2 = points2[i][0]
        v2 = points2[i][1]
        W[i] = np.array([u1*u2, u2*v1, u2, v2*u1, v1*v2, v2, u1, v1, 1])

    # compute F_hat
    U, s, VT = np.linalg.svd(W, full_matrices=True)
    f = VT[-1, :]
    F_hat = np.reshape(f, (3, 3))

    # compute F
    U, s_hat, VT = np.linalg.svd(F_hat, full_matrices=True)
    s = np.zeros((3, 3))
    s[0][0] = s_hat[0]
    s[1][1] = s_hat[1]
    F = np.dot(U, np.dot(s, VT))

    return F

'''
NORMALIZED_EIGHT_POINT_ALG calcule la matrice fondamentale à partir des points de correspondance
à l'aide de l'algorithme normalisé à huit points
Arguments :
    points1 - N points de la première image qui correspondent aux points2
    points2 - N points de la deuxième image qui correspondent aux points1

    Les points 1 et 2 proviennent de la méthode get_data_from_txt_file()
Retourne :
    F - la matrice fondamentale telle que (points2)^T * F * points1 = 0

'''
def normalized_eight_point_alg(points1, points2):
    N = points1.shape[0]
    points1_uv = points1[:, 0:2]
    points2_uv = points2[:, 0:2]

    # normalization
    mean1 = np.mean(points1_uv, axis=0)
    mean2 = np.mean(points2_uv, axis=0)

    points1_center = points1_uv - mean1
    points2_center = points2_uv - mean2

    scale1 = np.sqrt(2/(np.sum(points1_center**2)/N * 1.0))
    scale2 = np.sqrt(2/(np.sum(points2_center**2)/N * 1.0))

    T1 = np.array([[scale1, 0, -mean1[0] * scale1],
                   [0, scale1, -mean1[1] * scale1],
                   [0, 0, 1]])

    T2 = np.array([[scale2, 0, -mean2[0] * scale2],
                   [0, scale2, -mean2[0] * scale2],
                   [0, 0, 1]])

    q1 = T1.dot(points1.T).T; q2 = T2.dot(points2.T).T
    Fq = lls_eight_point_alg(q1, q2)
    F = T2.T.dot(Fq).dot(T1)

    return F

'''
PLOT_EPIPOLAR_LINES_ON_IMAGES étant donné une paire d'images et les points correspondants,
trace les lignes épipolaires sur les images
Arguments :
    points1 - N points de la première image qui correspondent aux points2
    points2 - N points de la seconde image qui correspondent aux points1
    im1 - une matrice HxW(xC) contenant les valeurs des pixels de la première image 
    im2 - matrice HxW(xC) contenant les valeurs des pixels de la seconde image 
    F - la matrice fondamentale telle que (points2)^T * F * points1 = 0

    Les points 1 et 2 proviennent de la méthode get_data_from_t_file()
Retourne :
    Rien ;
'''
def plot_epipolar_lines_on_images(points1, points2, im1, im2, F):
    plt.subplot(1,2,1)
    ln1 = F.T.dot(points2.T)
    for i in range(ln1.shape[1]):
        plt.plot([0, im1.shape[1]], [-ln1[2][i]*1.0/ln1[1][i], -(ln1[2][i]+ln1[0][i]*im1.shape[1])*1.0/ln1[1][i]], 'r')
        plt.plot([points1[i][0]], [points1[i][1]], 'b*')
    plt.imshow(im1, cmap='gray')

    plt.subplot(1,2,2)
    ln2 = F.dot(points1.T)
    for i in range(ln2.shape[1]):
        plt.plot([0, im2.shape[1]], [-ln2[2][i]*1.0/ln2[1][i], -(ln2[2][i]+ln2[0][i]*im2.shape[1])/ln2[1][i]], 'r')
        plt.plot([points2[i][0]], [points2[i][1]], 'b*')
    plt.imshow(im2, cmap='gray')

'''
COMPUTE_DISTANCE_TO_EPIPOLAR_LINES calcule la distance moyenne d'un ensemble de points à leurs lignes épipolaires correspondantes. 
points à leurs lignes épipolaires correspondantes
Arguments :
    points1 - N points de la première image qui correspondent aux points2
    points2 - N points de la seconde image qui correspondent aux points1
    F - la matrice fondamentale telle que (points2)^T * F * points1 = 0

    Les points1 et 2 proviennent de la méthode get_data_from_t_file()
Retourne :
    average_distance - la distance moyenne de chaque point à la ligne épipolaire
'''
def compute_distance_to_epipolar_lines(points1, points2, F):
    l = F.T.dot(points2.T)
    dist_sum = 0.0
    points_num = points1.shape[0]

    for i in range(points_num):
        dist_sum += np.abs(points1[i][0]*l[0][i] + points1[i][1]*l[1][i] + l[2][i]) * 1.0 \
                    / np.sqrt(l[0][i]**2 + l[1][i]**2)
    return dist_sum / points_num

