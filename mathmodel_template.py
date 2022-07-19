# -*- coding: UTF-8 -*-

import numpy as np

# =========================================================
# A template of hollow sphere math model
# with various sld equal to the radius of certain point
# =========================================================

# =========================== ! ===========================
# Do not change the class name, attributes name or method name !
# =========================================================
class SpecificMathModel:
    '''to generate a 3D model from a mathematical description
    for example: a hollow sphere is "x**2+y**2+z**2 >= R1**2 and x**2+y**2+z**2 <= R2**2
    also, in spherical coordinates, a hollow sphere is r >=R1 and r <= R2

    coord:
    - 'car' |in (x, y, z)
    - 'sph' |in (r, theta, phi) |theta: 0~2pi ; phi: 0~pi
    - 'cyl' |in (rho, phi, z) |theta:0-2pi
    '''
    def __init__(self):
        '''must at least have these 2 attributes
        '''
        self.params = {
            'R1': 8,
            'R2': 10
        }
        self.coord = 'sph'  # 'car' or 'sph' or 'cyl'

    def get_boundary(self):
        '''re-generate boundary in case that params are altered in software
        '''
        boundary_min = -self.params['R2']*np.ones(3)
        boundary_max = self.params['R2']*np.ones(3)
        return boundary_min, boundary_max

    def sld(self, grid_points_in_coord):
        ''' calculate sld values of each grid points
        Args:
            grid_points_in_coord: ndarray, shape == (n, 3), value is in the coordinates of self.coord
        Returns:
            sld: ndarray, shape == (n,) indicates the sld value of each grid points 
        '''
        points_sph = grid_points_in_coord
        r, theta, phi = points_sph[:,0], points_sph[:,1], points_sph[:,2]
        R1 = self.params['R1']
        R2 = self.params['R2']

        index = np.zeros_like(r)
        index[(r>=R1) & (r<=R2)] = 1
        sld = r**2 * index
        return sld