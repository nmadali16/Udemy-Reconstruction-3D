import sys
import numpy as np
import os
import scipy.misc
from scipy.optimize import least_squares
import math
from copy import deepcopy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sfm_utils import *

'''
ESTIMATE_INITIAL_RT de la matrice essentielle, nous pouvons calculer 4 estimations initiales de la RT relative entre les deux caméras.
de la RT relative entre les deux caméras
Arguments :
    E - la matrice essentielle entre les deux caméras
Retourne :
    RT : un tenseur 4x3x4 dans lequel la matrice 3x4 RT[i, :,:] est l'une des
        quatre transformations possibles
'''
def estimate_initial_RT(E):
    U, s, VT = np.linalg.svd(E)
    # compute R
    Z = np.array([[0, 1, 0],
                  [-1, 0, 0],
                  [0, 0, 0]])
    W = np.array([[0, -1, 0],
                  [1, 0, 0],
                  [0, 0, 1]])

    M = U.dot(Z).dot(U.T)
    Q1 = U.dot(W).dot(VT)
    R1 = np.linalg.det(Q1) * 1.0 * Q1

    Q2 = U.dot(W.T).dot(VT)
    R2 = np.linalg.det(Q2) * 1.0 * Q2

    # compute T
    T1 = U[:, 2].reshape(-1, 1)
    T2 = -U[:, 2].reshape(-1, 1)

    R_set = [R1, R2]
    T_set = [T1, T2]
    RT_set = []
    for i in range(len(R_set)):
        for j in range(len(T_set)):
            RT_set.append(np.hstack((R_set[i], T_set[j])))

    RT = np.zeros((4, 3, 4))
    for i in range(RT.shape[0]):
        RT[i, :, :] = RT_set[i]

    return RT

'''
LINEAR_ESTIMATE_3D_POINT étant donné un point correspondant dans différentes images,
calcule le point 3D qui est la meilleure estimation linéaire
Arguments :
    image_points - les points mesurés dans chacune des M images (matrice Mx2)
    camera_matrices - les matrices projectives de la caméra (tenseur Mx3x4)
Retourne :
    point_3d - le point 3D
'''
def linear_estimate_3d_point(image_points, camera_matrices):
    N = image_points.shape[0]
    A = np.zeros((2*N, 4))
    A_set = []

    for i in range(N):
        pi = image_points[i]
        Mi = camera_matrices[i]
        Aix = pi[0]*Mi[2] - Mi[0]
        Aiy = pi[1]*Mi[2] - Mi[1]
        A_set.append(Aix)
        A_set.append(Aiy)

    for i in range(A.shape[0]):
        A[i] = A_set[i]

    U, s, VT = np.linalg.svd(A)
    P_homo = VT[-1]
    P_homo /= P_homo[-1]
    P = P_homo[:3]

    return P
'''
REPROJECTION_ERROR Étant donné un point 3D et ses points correspondants dans les plans de l'image, calculer le vecteur d'erreur de reprojection et le jacobien associé.
calcule le vecteur d'erreur de reprojection et le jacobien associé.
Arguments :
    point_3d - le point 3D correspondant aux points de l'image
    image_points - les points mesurés dans chacune des M images (matrice Mx2)
    camera_matrices - les matrices projectives des caméras (tenseur Mx3x4)
Retourne :
    error - le vecteur d'erreur de reprojection 2Mx1
'''
def reprojection_error(point_3d, image_points, camera_matrices):
    N = image_points.shape[0]
    error_set = []
    point_3d_homo = np.hstack((point_3d, 1))

    for i in range(N):
        pi = image_points[i]
        Mi = camera_matrices[i]
        Yi = Mi.dot(point_3d_homo)
        # compute error
        pi_prime = 1.0 / Yi[2] * np.array([Yi[0], Yi[1]])
        error_i = (pi_prime - pi)
        error_set.append(error_i[0])
        error_set.append(error_i[1])

    error = np.array(error_set)
    return error
'''
JACOBIAN Étant donné un point 3D et ses points correspondants dans les plans de l'image, calculer le vecteur d'erreur de reprojection et le jacobien associé.
calculer le vecteur d'erreur de reprojection et le jacobien associé
Arguments :
    point_3d - le point 3D correspondant aux points de l'image
    camera_matrices - les matrices projectives de la caméra (tenseur Mx3x4)
Retourne :
    jacobian - la matrice jacobienne 2Mx3
'''
def jacobian(point_3d, camera_matrices):
    J = np.zeros((2*camera_matrices.shape[0], 3))
    point_3d_homo = np.hstack((point_3d, 1))
    J_set = []

    for i in range(camera_matrices.shape[0]):
        Mi = camera_matrices[i]
        pi = Mi.dot(point_3d_homo)
        Jix = (pi[2]*np.array([Mi[0, 0], Mi[0, 1], Mi[0, 2]]) \
              - pi[0]*np.array([Mi[2, 0], Mi[2, 1], Mi[2, 2]])) / pi[2]**2
        Jiy = (pi[2]*np.array([Mi[1, 0], Mi[1, 1], Mi[1, 2]]) \
              - pi[1]*np.array([Mi[2, 0], Mi[2, 1], Mi[2, 2]])) / pi[2]**2
        J_set.append(Jix)
        J_set.append(Jiy)

    for i in range(J.shape[0]):
        J[i] = J_set[i]
    return J

'''
NONLINEAR_ESTIMATE_3D_POINT étant donné des points correspondants dans différentes images,
calcule le point 3D qui met à jour itérativement les points
Arguments :
    image_points - les points mesurés dans chacune des M images (matrice Mx2)
    camera_matrices - les matrices projectives de la caméra (tenseur Mx3x4)
Retourne :
    point_3d - le point 3D
'''
def nonlinear_estimate_3d_point(image_points, camera_matrices):
    P = linear_estimate_3d_point(image_points, camera_matrices)

    for i in range(10):
        e = reprojection_error(P, image_points, camera_matrices)
        J = jacobian(P, camera_matrices)
        P -= np.linalg.inv(J.T.dot(J)).dot(J.T).dot(e)

    return P

'''
ESTIMATE_RT_FROM_E de la matrice essentielle, nous pouvons calculer la RT relative entre les deux caméras. 
relative entre les deux caméras
Arguments :
    E - la matrice essentielle entre les deux caméras
    image_points - N points mesurés dans chacune des M images (matrice NxMx2)
    K - la matrice intrinsèque de la caméra
Retourne :
    RT : La matrice 3x4 qui donne la rotation et la translation entre les 
        deux caméras
'''
def estimate_RT_from_E(E, image_points, K):
    RT = estimate_initial_RT(E)
    count = np.zeros((1, 4))
    I0 = np.array([[1.0, 0.0, 0.0, 0.0],
                   [0.0, 1.0, 0.0, 0.0],
                   [0.0, 0.0, 1.0, 0.0]])
    M1 = K.dot(I0)

    camera_matrices = np.zeros((2, 3, 4))
    camera_matrices[0] = M1
    for i in range(RT.shape[0]):
        RTi = RT[i] # 3x4 matrix
        M2i = K.dot(RTi)
        camera_matrices[1] = M2i
        for j in range(image_points.shape[0]):
            pointj_3d = nonlinear_estimate_3d_point(image_points[j], camera_matrices)
            Pj = np.vstack((pointj_3d.reshape(3, 1), [1]))
            Pj_prime = camera1tocamera2(Pj, RTi)
            if Pj[2] > 0 and Pj_prime[2] > 0:
                count[0, i] += 1

    maxIndex = np.argmax(count)
    maxRT = RT[maxIndex]
    return maxRT

def camera1tocamera2(P, RT):
    P_homo = np.array([P[0], P[1], P[2], 1.0])
    A = np.zeros((4, 4))
    A[0:3, :] = RT
    A[3, :] = np.array([0.0, 0.0, 0.0, 1.0])
    P_prime_homo = A.dot(P_homo.T)
    P_prime_homo /= P_prime_homo[3]
    P_prime = P_prime_homo[0:3]
    return P_prime
