# -*- coding: UTF-8 -*-

import numpy as np


class specific_mathmodel:

    def __init__(self):
        self.params = {
            'R': 30,
            't1': 5,
            't2': 15,
            'sld_core': 1,
            'sld_t1': 3,
            'sld_t2': 2
        }
        R_overall = self.params['R'] + self.params['t1'] + self.params['t2']
        self.boundary_min = -R_overall*np.ones(3)
        self.boundary_max = R_overall*np.ones(3)
        self.coord = 'sph'  # 'xyz' or 'sph' or 'cyl'

    def shape(self, grid_in_coord):
        points_sph = grid_in_coord
        self.points_sph = points_sph
        R = self.params['R']
        t1 = self.params['t1']
        t2 = self.params['t2']
        R_overall = R + t1 + t2

        #in_model_grid_index = np.zeros(points_sph.shape[0])
        r = points_sph[:, 0]
        in_model_grid_index = np.sign(np.sign(R_overall - r) + 1)
        self.in_model_grid_index = in_model_grid_index
        return self.in_model_grid_index

    def sld(self):
        points_sph = self.points_sph
        R = self.params['R']
        t1 = self.params['t1']
        t2 = self.params['t2']
        sld_core = self.params['sld_core']
        sld_t1 = self.params['sld_t1']
        sld_t2 = self.params['sld_t2']

        r = points_sph[:, 0]
        sld_index = np.zeros_like(r)
        # core
        sld_index[r<=R] = sld_core
        # t1
        sld_index[(r>R) & (r<=R+t1)] = sld_t1
        # t2
        sld_index[(r>R+t1) & (r<=R+t1+t2)] = sld_t2

        return sld_index