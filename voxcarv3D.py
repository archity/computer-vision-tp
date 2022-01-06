import math as m
import matplotlib.image as mpimg
import numpy as np
import skimage
from skimage import measure
import torch
from torch import nn
import trimesh


# Camera Calibration for Al's image[1..12].pgm   
calib=np.array([[-230.924, 0, -33.6163, 300,  -78.8596, -178.763, -127.597, 300,  -0.525731, 0, -0.85065, 2],
[-178.763, -127.597, -78.8596, 300,  0, -221.578, 73.2053, 300,  0, -0.85065, -0.525731, 2],
[-73.2053, 0, -221.578, 300,  78.8596, -178.763, -127.597, 300,  0.525731, 0, -0.85065, 2],
[-178.763, 127.597, -78.8596, 300,  0, 33.6163, -230.924, 300,  0, 0.85065, -0.525731, 2],
[73.2053, 0, 221.578, 300,  -78.8596, -178.763, 127.597, 300,  -0.525731, 0, 0.85065, 2],
[230.924, 0, 33.6163, 300,  78.8596, -178.763, 127.597, 300,  0.525731, 0, 0.85065, 2],
[178.763, -127.597, 78.8596, 300,  0, -221.578, -73.2053, 300,  0, -0.85065, 0.525731, 2],
[178.763, 127.597, 78.8596, 300,  0, 33.6163, 230.924, 300,  0, 0.85065, 0.525731, 2],
[-127.597, -78.8596, 178.763, 300,  -33.6163, -230.924, 0, 300,  -0.85065, -0.525731, 0, 2],
[-127.597, 78.8596, 178.763, 300,  -221.578, -73.2053, 0, 300,  -0.85065, 0.525731, 0, 2],
[127.597, 78.8596, -178.763, 300,  221.578, -73.2053, 0, 300,  0.85065, 0.525731, 0, 2],
[127.597, -78.8596, -178.763, 300,  33.6163, -230.924, 0, 300,  0.85065, -0.525731, 0, 2]])


# Build 3D grids
resolution=300   # 3D Grids are of size: resolution x resolution x resolution/2
step = 2/resolution    
X, Y, Z = np.mgrid[-1:1:step, -1:1:step, -0.5:0.5:step]  # Voxel coordinates
occupancy = np.ndarray(shape=(resolution,resolution,int(resolution/2)), dtype=int) #Voxel occupancy
occupancy.fill(1) # Voxels are initially occupied then carved with silhouette information
 

####### MAIN #########  
if __name__ == "__main__":
    
    i=1
    myFile = "image{0}.pgm".format(i) # read the input silhouettes
    print(myFile)
    img = mpimg.imread(myFile)
    if img.dtype == np.float32: # if not integer
        img = (img * 255).astype(np.uint8)

    # Compute grid projection in images
    #TO BE COMPLETED


    
    # Update grid occupancy
    #TO BE COMPLETED

    # Voxel visualization
    verts, faces, normals, values = measure.marching_cubes(occupancy, 0.25) # Marching cubes 
    surf_mesh = trimesh.Trimesh(verts, faces, validate=True) # Export in a standard file format 
    surf_mesh.export('alvoxels.off')
 
