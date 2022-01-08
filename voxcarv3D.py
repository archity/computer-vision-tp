import math as m
import matplotlib.image as mpimg
import numpy as np
import skimage
from skimage import measure
import torch
from torch import nn
import trimesh

# Camera Calibration for Al's image[1..12].pgm
calib = np.array([[-230.924, 0, -33.6163, 300, -78.8596, -178.763, -127.597, 300, -0.525731, 0, -0.85065, 2],
                  [-178.763, -127.597, -78.8596, 300, 0, -221.578, 73.2053, 300, 0, -0.85065, -0.525731, 2],
                  [-73.2053, 0, -221.578, 300, 78.8596, -178.763, -127.597, 300, 0.525731, 0, -0.85065, 2],
                  [-178.763, 127.597, -78.8596, 300, 0, 33.6163, -230.924, 300, 0, 0.85065, -0.525731, 2],
                  [73.2053, 0, 221.578, 300, -78.8596, -178.763, 127.597, 300, -0.525731, 0, 0.85065, 2],
                  [230.924, 0, 33.6163, 300, 78.8596, -178.763, 127.597, 300, 0.525731, 0, 0.85065, 2],
                  [178.763, -127.597, 78.8596, 300, 0, -221.578, -73.2053, 300, 0, -0.85065, 0.525731, 2],
                  [178.763, 127.597, 78.8596, 300, 0, 33.6163, 230.924, 300, 0, 0.85065, 0.525731, 2],
                  [-127.597, -78.8596, 178.763, 300, -33.6163, -230.924, 0, 300, -0.85065, -0.525731, 0, 2],
                  [-127.597, 78.8596, 178.763, 300, -221.578, -73.2053, 0, 300, -0.85065, 0.525731, 0, 2],
                  [127.597, 78.8596, -178.763, 300, 221.578, -73.2053, 0, 300, 0.85065, 0.525731, 0, 2],
                  [127.597, -78.8596, -178.763, 300, 33.6163, -230.924, 0, 300, 0.85065, -0.525731, 0, 2]])

# Build 3D grids
resolution = 300  # 3D Grids are of size: resolution x resolution x resolution/2
step = 2 / resolution
X, Y, Z = np.mgrid[-1:1:step, -1:1:step, -0.5:0.5:step]  # Voxel coordinates
occupancy = np.ndarray(shape=(resolution, resolution, int(resolution / 2)), dtype=int)  # Voxel occupancy
occupancy.fill(1)  # Voxels are initially occupied then carved with silhouette information


def compute_grid_non_vec(occupancy):
    """
    Non-vectorized solution
    :return:
    """
    i = 1
    myFile = "./img/image{0}.pgm".format(i)  # read the input silhouettes
    print(myFile)
    img = mpimg.imread(myFile)
    if img.dtype == np.float32:  # if not integer
        img = (img * 255).astype(np.uint8)

    for x in range(X.shape[0]):
        for y in range(X.shape[1]):
            for i in range(X.shape[2]):
                result = np.dot(calib[0].reshape(3, 4), np.array([X[x][y][i], Y[x][y][i], Z[x][y][i], 1]))
                u = int(result[0] / result[2])
                v = int(result[1] / result[2])
                if u >= 300 or v >= 300:
                    u = 299
                    v = 299
                elif img[u][v] == 0:
                    # Update grid occupancy
                    occupancy[x][y][i] = 0

    # Voxel visualization
    voxel_visualization(occupancy, i)


def compute_grid_vec(occupancy):
    """
    Edmond's vectorized solution
    :return: Modified voxel occupancy
    """
    # For each image
    for i in range(12):
        myFile = f"./img/image{i}.pgm"  # read the input silhouettes
        print(myFile)
        img = mpimg.imread(myFile)
        if img.dtype == np.float32:  # if not integer
            img = (img * 255).astype(np.uint8)

        w = calib[i][8] * X + calib[i][9] * Y + calib[i][10] * Z + calib[i][11]
        u = calib[i][0] * X + calib[i][1] * Y + calib[i][2] * Z + calib[i][3] / w
        v = calib[i][4] * X + calib[i][5] * Y + calib[i][6] * Z + calib[i][7] / w
        u = u.astype(int)
        v = v.astype(int)
        u[u >= 300] = 299
        v[v >= 300] = 299
        u[u < 0] = 0
        v[v < 0] = 0

        # Update grid occupemcy
        img[img > 0] = 1
        occupancy = occupancy * img[v, u]

        # Voxel visualization
        voxel_visualization(occupancy, i)


def voxel_visualization(occupancy, file_num):
    """
    Voxel visualization: generate and save the visualizations
    :return:
    """

    verts, faces, normals, values = measure.marching_cubes(occupancy, 0.25)  # Marching cubes
    surf_mesh = trimesh.Trimesh(verts, faces, validate=True)  # Export in a standard file format
    surf_mesh.export(f"img/alvoxels{file_num}.off")
    print(f"Saved: alvoxels{file_num}.off")


if __name__ == "__main__":
    # print(img.shape)
    # print(calib.shape)
    # print(X.shape)
    # print(occupancy.shape)

    # Compute grid projection in images
    # compute_grid_non_vec(occupancy)
    compute_grid_vec(occupancy)

