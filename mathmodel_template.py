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
    for example: a spherical shell is "x**2+y**2+z**2 >= R_core**2 and x**2+y**2+z**2 <= (R_core+thickness)**2
    also, in spherical coordinates, a hollow sphere is (r >= R_core) and (r <= R_core+thickness)

    coord:
    - 'car' |in (x, y, z)
    - 'sph' |in (r, theta, phi) |theta: 0~2pi ; phi: 0~pi
    - 'cyl' |in (rho, phi, z) |theta:0-2pi
    '''
    def __init__(self) -> None:
        '''must at least have these 2 attributes
        '''
        self.params = {
            'R_core': 40,
            'thickness': 10
        }
        self.coord = 'sph'  # 'car' or 'sph' or 'cyl'

    def get_boundary(self) -> tuple:
        '''re-generate boundary for every method call
        in case that params are altered in software
        '''
        boundary_max = (self.params['R_core']+self.params['thickness']) * np.ones(3)
        boundary_min = -boundary_max
        return boundary_min, boundary_max

    def sld(self, grid_points_in_coord:np.ndarray) -> np.ndarray:
        ''' calculate sld values of each grid points
        Args:
            grid_points_in_coord: ndarray, shape == (n, 3), value is in the coordinates of self.coord
        Returns:
            sld: ndarray, shape == (n,) indicates the sld value of each grid points 
        '''
        points_sph = grid_points_in_coord
        r, theta, phi = points_sph[:,0], points_sph[:,1], points_sph[:,2]
        R = self.params['R_core']
        t = self.params['thickness']

        index = np.zeros_like(r)
        index[(r>=R) & (r<=(R+t))] = 1
        #grid_sld = r * np.cos(theta) * np.sin(phi) * index
        grid_sld = 1 * index
        return grid_sld