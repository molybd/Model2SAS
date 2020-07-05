# -*- coding: UTF-8 -*-

import numpy as np

# to generate a 3D model from a mathematical description
# for example: a hollow sphere is "x**2+y**2+z**2 >= R1**2 and x**2+y**2+z**2 <= R2**2
# also, in spherical coordinates, a hollow sphere is r >=R1 and r <= R2
#
# function recieve an array of n points, shape = (n, 3)
#
# function must return an array with shape = (n,)
# each figure in this array cooresponds to a point in the input array
# in the output array, 1 means the the point is in the model,
# 0 means the point is not in the model
#
# boundaryList is 
#   - [xmin, xmax, ymin, ymax, zmin, zmax]
# coord is 
#   - 'xyz' |in (x,y,z)
#   - 'sph' |in (r, theta, phi)|theta: 0~pi ; phi: 0~2pi
#   - 'cyl' |in (r, phi, z)|phi:0-2pi
# coord must be the last arguement



# function name must be model !
# do not change the keyword arguments named 'coord' and 'boundry_xyz' !
# boundry_xyz is a list in format [xmin, xmax, ymin, ymax, zmin, zmax]
def model(points_xyz, param1=30, param2=60, param3=90, boundary_xyz=[-100,100,-100,100,-100,100], coord='xyz'):
    pass
