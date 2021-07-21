# -*- coding: UTF-8 -*-

import numpy as np

'''
to generate a 3D model from a mathematical description
for example: a hollow sphere is "x**2+y**2+z**2 >= R1**2 and x**2+y**2+z**2 <= R2**2
also, in spherical coordinates, a hollow sphere is r >=R1 and r <= R2

coord is 
  - 'xyz' |in (x, y, z)
  - 'sph' |in (r, theta, phi) |theta: 0~2pi ; phi: 0~pi
  - 'cyl' |in (rho, phi, z) |theta:0-2pi
'''

# Don't change the class name, attributes name or method name !
class specific_mathmodel:
    '''A template of hollow sphere math model
    '''

    def __init__(self):
        self.params = {
            'R1': 10,
            'R2': 15
        }
        self.boundary_min = -self.params['R2']*np.ones(3)
        self.boundary_max = self.params['R2']*np.ones(3)
        self.coord = 'sph'  # 'xyz' or 'sph' or 'cyl'
        # must have these 4 attributes

    def shape(self, grid_in_coord):
        points_sph = grid_in_coord
        R1 = self.params['R1']
        R2 = self.params['R2']

        in_model_grid_index = np.zeros(points_sph.shape[0])
        r = points_sph[:, 0]
        R1_array = R1 * np.ones_like(r)
        R2_array = R2 * np.ones_like(r)
        in_model_grid_index = np.sign(r-R1_array) * np.sign(R2_array-r)
        in_model_grid_index = np.sign(in_model_grid_index+1)
        self.in_model_grid_index = in_model_grid_index
        return self.in_model_grid_index  # must return in_model_grid_index

    def sld(self):
        return 15 * self.in_model_grid_index  # must return sld index