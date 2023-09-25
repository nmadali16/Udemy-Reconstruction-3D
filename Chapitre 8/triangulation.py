import numpy as np
from utils import *


def TriangulateImage(K, image_name, images, registered_images, matches):

  
  # Loop over all registered images and triangulate new points with the new image.
  # Make sure to keep track of all new 2D-3D correspondences, also for the registered images

  image = images[image_name]
  points3D = np.zeros((0,3))
  # You can save the correspondences for each image in a dict and refer to the `local` new point indices here.
  # Afterwards you just add the index offset before adding the correspondences to the images.
  corrs = {}

  for reg_image in registered_images:
      save_index = len(points3D)
      current_image = images[reg_image]
      current_matches = GetPairMatches(image_name, reg_image, matches)
      curr_points3D, im1_corrs, im2_corrs = TriangulatePoints(K, image, current_image, current_matches)
      points3D = np.append(points3D, curr_points3D, axis = 0)

      index_corr_im1 = save_index + np.arange(im1_corrs.shape[0])
      index_corr_im2 = save_index + np.arange(im2_corrs.shape[0])

      corrs[image_name] = np.array([im1_corrs, index_corr_im1])
      corrs[reg_image] = np.array([im2_corrs, index_corr_im2])


  return points3D, corrs


# Find (unique) 2D-3D correspondences from 2D-2D correspondences
def Find2D3DCorrespondences(image_name, images, matches, registered_images):
  assert(image_name not in registered_images)

  image_kp_idxs = []
  p3D_idxs = []
  for other_image_name in registered_images:
    other_image = images[other_image_name]
    pair_matches = GetPairMatches(image_name, other_image_name, matches)

    for i in range(pair_matches.shape[0]):
      p3D_idx = other_image.GetPoint3DIdx(pair_matches[i,1])
      if p3D_idx > -1:
        p3D_idxs.append(p3D_idx)
        image_kp_idxs.append(pair_matches[i,0])

  print(f'found {len(p3D_idxs)} points, {np.unique(np.array(p3D_idxs)).shape[0]} unique points')

  # Remove duplicated correspondences
  _, unique_idxs = np.unique(np.array(p3D_idxs), return_index=True)
  image_kp_idxs = np.array(image_kp_idxs)[unique_idxs].tolist()
  p3D_idxs = np.array(p3D_idxs)[unique_idxs].tolist()

  return image_kp_idxs, p3D_idxs



def TriangulatePoints(K, im1, im2, matches):

  R1, t1 = im1.Pose()
  R2, t2 = im2.Pose()
  P1 = K @ np.append(R1, np.expand_dims(t1, 1), 1)
  P2 = K @ np.append(R2, np.expand_dims(t2, 1), 1)

  # Ignore matches that already have a triangulated point
  new_matches = np.zeros((0, 2), dtype=int)

  num_matches = matches.shape[0]
  for i in range(num_matches):
    p3d_idx1 = im1.GetPoint3DIdx(matches[i, 0])
    p3d_idx2 = im2.GetPoint3DIdx(matches[i, 1])
    if p3d_idx1 == -1 and p3d_idx2 == -1:
      new_matches = np.append(new_matches, matches[[i]], 0)


  num_new_matches = new_matches.shape[0]

  points3D = np.zeros((num_new_matches, 3))

  for i in range(num_new_matches):

    kp1 = im1.kps[new_matches[i, 0], :]
    kp2 = im2.kps[new_matches[i, 1], :]

    # H & Z Sec. 12.2
    A = np.array([
      kp1[0] * P1[2] - P1[0],
      kp1[1] * P1[2] - P1[1],
      kp2[0] * P2[2] - P2[0],
      kp2[1] * P2[2] - P2[1]
    ])

    _, _, vh = np.linalg.svd(A)
    homogeneous_point = vh[-1]
    points3D[i] = homogeneous_point[:-1] / homogeneous_point[-1]

  # We need to keep track of the correspondences between image points and 3D points
  im1_corrs = new_matches[:,0]
  im2_corrs = new_matches[:,1]

  # TODO
  # Filter points behind the cameras by transforming them into each camera space and checking the depth (Z)
  # Make sure to also remove the corresponding rows in `im1_corrs` and `im2_corrs`
  points3D_C1_transposed = R1 @ points3D.transpose() + t1[:,None]
  points3D_C1 = points3D_C1_transposed.transpose()

  points3D_C2_transposed = R2 @ points3D.transpose() + t2[:,None]
  points3D_C2 = points3D_C2_transposed.transpose()

  to_remove = []

  for point_index in range(np.shape(points3D)[0]):
     if(points3D_C1[point_index,2] < 0) or (points3D_C2[point_index,2] < 0):
         to_remove.append(point_index)

  points3D  = np.delete(points3D, to_remove, axis = 0)
  im1_corrs = np.delete(im1_corrs, to_remove)
  im2_corrs = np.delete(im2_corrs, to_remove)

  return points3D, im1_corrs, im2_corrs