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
    with various sld equal to the radius of certain point
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

    def getBoundary(self):
        # re-generate boundary in case that params are altered in software
        self.boundary_min = -self.params['R2']*np.ones(3)
        self.boundary_max = self.params['R2']*np.ones(3)
        return self.boundary_min, self.boundary_max

    def shape(self, grid_in_coord):
        points_sph = grid_in_coord
        self.points_sph = points_sph  # for usage in self.sld()
        R1 = self.params['R1']
        R2 = self.params['R2']

        r = points_sph[:, 0]
        in_model_grid_index = np.zeros_like(r)
        in_model_grid_index[(r>=R1) & (r<=R2)] = 1
        self.in_model_grid_index = in_model_grid_index
        return self.in_model_grid_index  # must return in_model_grid_index

    def sld(self):
        r = self.points_sph[:, 0]
        return r * self.in_model_grid_index  # must return sld index