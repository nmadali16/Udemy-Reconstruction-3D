import gc
import itertools
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os


class Image:
  # Constructor that reads the keypoints and image file from the data directory
  def __init__(self, data_folder, name):
      image_path = os.path.join(data_folder, 'images', name)
      keypoints_path = os.path.join(data_folder,'keypoints', name + '.txt')

      self.name = name
      self.image = plt.imread(image_path)
      self.kps = np.loadtxt(keypoints_path)
      self.p3D_idxs = {}

  # Set the image pose.
  # This is assumed to be the transformation from global space to image space
  def SetPose(self, R, t):
    self.R = R
    self.t = t

  # Get the image pose
  def Pose(self):
    return self.R, self.t

  # Add a new 2D-3D correspondence to the image
  # The function expects two equal length lists of indices, the first one with the
  # keypoint index in the image, the second one with the 3D point index in the reconstruction.
  def Add3DCorrs(self, kp_idxs, p3D_idxs):
    for corr in zip(kp_idxs, p3D_idxs):
      self.p3D_idxs[corr[0]] = corr[1]

  # Get the 3D point associated with the given keypoint.
  # Will return -1 if no correspondence is set for this keypoint.
  def GetPoint3DIdx(self, kp_idx):
    if kp_idx in self.p3D_idxs:
      return self.p3D_idxs[kp_idx]
    else:
      return -1

  # Get the number of 3D points that are observed by this image
  def NumObserved(self):
    return len(self.p3D_idxs)
# Update the reconstruction with the new information from a triangulated image
def UpdateReconstructionState(new_points3D, corrs, points3D, images):

  # TODO
  # Add the new points to the set of reconstruction points and add the correspondences to the images.
  # Be careful to update the point indices to the global indices in the `points3D` array.
  offset = len(points3D)
  points3D = np.append(points3D, new_points3D, 0)

  for im_name in corrs:

    images[im_name].Add3DCorrs(corrs[im_name][0], corrs[im_name][1]+offset)

  return points3D, images

def BuildProjectionConstraintMatrix(points2D, points3D):

  # TODO
  # For each correspondence, build the two rows of the constraint matrix and stack them

  num_corrs = points2D.shape[0]
  constraint_matrix = np.zeros((num_corrs * 2, 12))

  counter = 0
  for i in range(num_corrs):

     index_first_constraint = counter
     index_second_constraint = counter + 1
     x = points2D[i]
     X = points3D[i]
     first_constraint = np.array([0 , 0 , 0 , 0 ,-X[0], -X[1], -X[2], -1, x[1]*X[0], x[1]*X[1], x[1]*X[2], x[1]])

     second_constraint = np.array([X[0],X[1],X[2], 1 , 0 , 0 , 0 , 0 , -x[0] * X[0], -x[0] * X[1], -x[0] * X[2], -x[0]])
     # TODO Add your code here
     constraint_matrix[index_first_constraint] = first_constraint
     constraint_matrix[index_second_constraint] = second_constraint

     counter = counter + 2

  return constraint_matrix

def EstimateImagePose(points2D, points3D, K):

  
  # We use points in the normalized image plane.
  # This removes the 'K' factor from the projection matrix.
  # We don't normalize the 3D points here to keep the code simpler.
  points2D = MakeHomogeneous(points2D, ax = 1)
  normalized_points2D_transposed = np.linalg.inv(K) @ points2D.transpose()
  normalized_points2D = normalized_points2D_transposed.transpose()

  constraint_matrix = BuildProjectionConstraintMatrix(normalized_points2D, points3D)

  # We don't use optimization here since we would need to make sure to only optimize on the se(3) manifold
  # (the manifold of proper 3D poses). This is a bit too complicated right now.
  # Just DLT should give good enough results for this dataset.

  # Solve for the nullspace
  _, _, vh = np.linalg.svd(constraint_matrix)
  P_vec = vh[-1,:]
  P = np.reshape(P_vec, (3, 4), order='C')

  # Make sure we have a proper rotation
  u, s, vh = np.linalg.svd(P[:,:3])
  R = u @ vh

  if np.linalg.det(R) < 0:
    R *= -1

  _, _, vh = np.linalg.svd(P)
  C = np.copy(vh[-1,:])

  t = -R @ (C[:3] / C[3])

  return R, t



def MakeHomogeneous(pts, ax=0):
  assert(pts.ndim <= 2)
  assert(ax < pts.ndim)

  if pts.ndim == 2:
    num_pts = pts.shape[1-ax]
    if ax == 0:
      return np.append(pts, np.ones((1, num_pts)), ax)
    else:
      return np.append(pts, np.ones((num_pts, 1)), ax)

  else:
    return np.append(pts, 1)


def HNormalize(pts, ax=0):
  assert(pts.ndim <= 2)
  assert(ax < pts.ndim)

  if pts.ndim == 2:
    if ax == 0:
      return pts[:-1,:] / pts[[-1], :]
    else:
      return pts[:,:-1] / pts[:, [-1]]

  else:
    return pts[:-1] / pts[-1]


def GetPairMatches(im1, im2, matches):
  if im1 < im2:
    return matches[(im1, im2)]
  else:
    return np.flip(matches[(im2, im1)], 1)
  
def ReadFeatureMatches(image_pair, data_folder):
  im1 = image_pair[0]
  im2 = image_pair[1]
  assert(im1 < im2)
  matchfile_path = os.path.join(data_folder, 'matches', im1 + '-' + im2 + '.txt')
  pair_matches = np.loadtxt(matchfile_path, dtype=int)
  return pair_matches

def ReadKMatrix(data_folder):
  path = os.path.join(data_folder, 'images', 'K.txt')
  K = np.loadtxt(path)
  return K