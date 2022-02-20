# -*- coding: UTF-8 -*-

import numpy as np

class specific_mathmodel:

    def __init__(self):
        self.params = {
            'radius': 30,
            'height': 30,
            'sld': 1
        }
        self.boundary_min = -self.params['radius']*np.ones(3)
        self.boundary_min[-1] = -self.params['height']/2
        self.boundary_max = -1 * self.boundary_min
        self.coord = 'cyl'  # 'xyz' or 'sph' or 'cyl'

    def shape(self, grid_in_coord):
        rho, theta, z = grid_in_coord[:,0], grid_in_coord[:,1], grid_in_coord[:,2]
        R = self.params['radius']
        H = self.params['height']

        in_model_grid_index = np.zeros_like(rho)
        in_model_grid_index[(rho<=R) & (np.abs(z)<=H/2)] = 1

        self.in_model_grid_index = in_model_grid_index
        return in_model_grid_index

    def sld(self):
        return self.params['sld']*self.in_model_grid_index