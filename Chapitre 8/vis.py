



import math
import matplotlib.pyplot as plt
import numpy as np





def PlotCamera(R, t, ax=None, scale=1.0, color='b'):
  if ax == None:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

  Rcw = R.transpose()
  tcw = -Rcw @ t

  # Define a path along the camera gridlines
  camera_points = np.array([
    [0, 0, 0],
    [1, 1, 1],
    [1, -1, 1],
    [-1, -1, 1],
    [-1, 1, 1],
    [1, 1, 1],
    [0, 0, 0],
    [1, -1, 1],
    [0, 0, 0],
    [-1, -1, 1],
    [0, 0, 0],
    [-1, 1, 1]
  ])

  # Make sure that this vector has the right shape
  tcw = np.reshape(tcw, (3, 1))

  cam_points_world = (Rcw @ (scale * camera_points.transpose()) + np.tile(tcw, (1, 12))).transpose()

  ax.plot(xs=cam_points_world[:,0], ys=cam_points_world[:,1], zs=cam_points_world[:,2], color=color)

  plt.show(block=False)

  return ax


def PlotImages(images):
  num_images = len(images)

  grid_height = math.floor(math.sqrt(num_images))
  grid_width = math.ceil(num_images / grid_height)

  fig = plt.figure()

  for idx, image_name in enumerate(images):
    ax = fig.add_subplot(grid_height, grid_width, idx+1)
    ax.imshow(images[image_name].image)
    ax.axis('off')

  plt.show(block=False)


def PlotWithKeypoints(image):
  fig = plt.figure()
  ax = fig.add_subplot(1,1,1)
  ax.imshow(image.image)
  ax.plot(image.kps[:,0], image.kps[:,1], 'r.')
  ax.axis('off')
  plt.show(block=False)


def PlotImagePairMatches(im1, im2, matches):
  pair_image_width = im1.image.shape[1] + im2.image.shape[1]
  pair_image_height = max(im1.image.shape[0], im2.image.shape[0])

  pair_image = np.ones((pair_image_height, pair_image_width, 3))

  im2_offset = im1.image.shape[1]

  pair_image[0:im1.image.shape[0], 0:im1.image.shape[1], :] = im1.image
  pair_image[0:im2.image.shape[0], im2_offset : im2.image.shape[1] + im2_offset] = im2.image

  fig = plt.figure()
  ax = fig.add_subplot(111)
  ax.imshow(pair_image)
  ax.plot(im1.kps[:,0], im1.kps[:,1], 'r.')
  ax.plot(im2.kps[:,0] + im2_offset, im2.kps[:,1], 'r.')
  for i in range(matches.shape[0]):
    kp1 = im1.kps[matches[i,0]]
    kp2 = im2.kps[matches[i,1]] 
    ax.plot([kp1[0], kp2[0] + im2_offset], [kp1[1], kp2[1]], 'g-', linewidth=1)
  ax.axis('off')
  ax.set_title(f'{im1.name} - {im2.name} ({matches.shape[0]})')
  plt.show()
  plt.close(fig)


def PlotCameras(images, registered_images, ax=None):

  for image_name in registered_images:
    image = images[image_name]
    R, t = image.Pose()
    ax = PlotCamera(R, t, ax, 0.5)

def Plot3DPoints(points, ax=None):
  if ax == None:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
  ax.plot(xs=points[:,0], ys=points[:,1], zs=points[:,2], color='k', marker='.', linestyle='None')
  ax.set_title('3D Scene')
  
  plt.show(block=False)

  return ax
  

def PlotCamera(R, t, ax=None, scale=1.0, color='b'):
  if ax == None:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

  Rcw = R.transpose()
  tcw = -Rcw @ t

  # Define a path along the camera gridlines
  camera_points = np.array([
    [0, 0, 0],
    [1, 1, 1],
    [1, -1, 1],
    [-1, -1, 1],
    [-1, 1, 1],
    [1, 1, 1],
    [0, 0, 0],
    [1, -1, 1],
    [0, 0, 0],
    [-1, -1, 1],
    [0, 0, 0],
    [-1, 1, 1]
  ])

  # Make sure that this vector has the right shape
  tcw = np.reshape(tcw, (3, 1))

  cam_points_world = (Rcw @ (scale * camera_points.transpose()) + np.tile(tcw, (1, 12))).transpose()

  ax.plot(xs=cam_points_world[:,0], ys=cam_points_world[:,1], zs=cam_points_world[:,2], color=color)

  plt.show(block=False)

  return ax


def Plot2DPoints(points, image_size, ax=None):
  if ax == None:
    fig = plt.figure()
    ax = fig.add_subplot(111)

  image_frame_points = np.array([
    [0, 0],
    [image_size[0], 0],
    [image_size[0], image_size[1]],
    [0, image_size[1]],
    [0, 0]
  ])
  ax.plot(image_frame_points[:,0], image_frame_points[:,1])
  ax.plot(points[:,0], image_size[1] - points[:,1], 'k.')
  ax.axis('equal')
  ax.set_title('Image')

  plt.show(block=False)


def PlotProjectedPoints(points3D, points2D, K, R, t, image_size, ax=None):
  if ax == None:
    fig = plt.figure()
    ax = fig.add_subplot(111)

  # Project points onto image
  p2d = K @ ((R @ points3D.transpose()) + t)
  p2d = p2d[0:2, :] / p2d[[-1],:]

  ax.plot(p2d[0,:], image_size[1] - p2d[1,:], 'r.')
  num_points = points2D.shape[0]
  for i in range(num_points):
    ax.plot([p2d[0,i], points2D[i,0]], [image_size[1] - p2d[1,i], image_size[1] - points2D[i,1]], color='g')

  plt.show(block=False)