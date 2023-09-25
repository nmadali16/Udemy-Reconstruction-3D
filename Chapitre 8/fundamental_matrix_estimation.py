import numpy as np


from utils import *


def DecomposeEssentialMatrix(E):

  u, s, vh = np.linalg.svd(E)

  # Determine the translation up to sign
  t_hat = u[:,-1]

  W = np.array([
    [0, -1, 0],
    [1, 0, 0],
    [0, 0, 1]
  ])

  # Compute the two possible rotations
  R1 = u @ W @ vh
  R2 = u @ W.transpose() @ vh

  # Make sure the orthogonal matrices are proper rotations
  if np.linalg.det(R1) < 0:
    R1 *= -1

  if np.linalg.det(R2) < 0:
    R2 *= -1

  # Assemble the four possible solutions
  sols = [
    (R1, t_hat),
    (R2, t_hat),
    (R1, -t_hat),
    (R2, -t_hat)
  ]

  return sols

def EstimateEssentialMatrix(K, im1, im2, matches):
  
  # Normalize coordinates (to points on the normalized image plane)

  kps1 = MakeHomogeneous(im1.kps, ax = 1)
  kps2 = MakeHomogeneous(im2.kps, ax = 1)

  # These are the keypoints on the normalized image plane (not to be confused with the normalization in the calibration exercise)
  normalized_kps1_transposed = np.linalg.inv(K) @ np.transpose(kps1)
  normalized_kps2_transposed = np.linalg.inv(K) @ np.transpose(kps2)

  normalized_kps1 = np.transpose(normalized_kps1_transposed)
  normalized_kps2 = np.transpose(normalized_kps2_transposed)


  
  # Assemble constraint matrix
  constraint_matrix = np.zeros((matches.shape[0], 9))

  for i in range(matches.shape[0]):
    
    # Add the constraints
      x1 = normalized_kps1[matches[i,0],:]
      x2 = normalized_kps2[matches[i,1],:]
      constraint_matrix[i] = np.array([x2[0] * x1[0] , x1[0] * x2[1], x1[0], x2[0] * x1[1], x1[1] * x2[1], x1[1], x2[0], x2[1], 1 ])


  # Solve for the nullspace of the constraint matrix
  _, _, vh = np.linalg.svd(constraint_matrix)
  vectorized_E_hat = vh[-1,:]

  
  # Reshape the vectorized matrix to it's proper shape again
  row_1, row_2, row_3 = np.split(vectorized_E_hat, 3)
  E_hat = np.array([row_1,row_2,row_3])

  
  # We need to fulfill the internal constraints of E
  # The first two singular values need to be equal, the third one zero.
  # Since E is up to scale, we can choose the two equal singluar values arbitrarily
  u, s, vh = np.linalg.svd(E_hat)
  singular_values = [1 ,1 ,0]
  E = u @ np.diag(singular_values) @ vh

  # This is just a quick test that should tell you if your estimated matrix is not correct
  # It might fail if you estimated E in the other direction (i.e. kp2' * E * kp1)
  # You can adapt it to your assumptions.
  for i in range(matches.shape[0]):
    kp1 = normalized_kps1[matches[i,0],:]
    kp2 = normalized_kps2[matches[i,1],:]

    #assert(abs(kp1.transpose() @ E @ kp2) < 0.01)

  return E

