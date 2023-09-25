import numpy as np
import math

def mat2euler(M, cy_thresh=None):
    ''' Découvrir le vecteur de l'angle d'Euler à partir d'une matrice 3x3

    Utilise les conventions ci-dessus.

    Paramètres
    ----------
    M : matrice, forme (3,3)
    cy_thresh : Aucun ou scalaire, optionnel
       seuil en dessous duquel il faut renoncer à l'arctan simple pour
       l'estimation de la rotation x.  S'il est nul (par défaut), l'estimation se fait à partir de la précision de l'entrée.
       de la précision de l'entrée.

    Retourne
    -------
    z : scalaire
    y : scalaire
    x : scalaire
       Rotations en radians autour des axes z, y, x, respectivement

    
    '''
    _FLOAT_EPS_4 = np.finfo(float).eps * 4.0
    M = np.asarray(M)
    if cy_thresh is None:
        try:
            cy_thresh = np.finfo(M.dtype).eps * 4
        except ValueError:
            cy_thresh = _FLOAT_EPS_4
    r11, r12, r13, r21, r22, r23, r31, r32, r33 = M.flat
    # cy: sqrt((cos(y)*cos(z))**2 + (cos(x)*cos(y))**2)
    cy = math.sqrt(r33*r33 + r23*r23)
    if cy > cy_thresh: # cos(y) not close to zero, standard form
        z = math.atan2(-r12,  r11) # atan2(cos(y)*sin(z), cos(y)*cos(z))
        y = math.atan2(r13,  cy) # atan2(sin(y), cy)
        x = math.atan2(-r23, r33) # atan2(cos(y)*sin(x), cos(x)*cos(y))
    else: # cos(y) (close to) zero, so x -> 0.0 (see above)
        # so r21 -> sin(z), r22 -> cos(z) and
        z = math.atan2(r21,  r22)
        y = math.atan2(r13,  cy) # atan2(sin(y), cy)
        x = 0.0
    return z, y, x
