# -*- coding: UTF-8 -*-

import torch
from torch import Tensor

# =========================================================
# A template of hollow sphere math model
# with various sld equal to the radius of certain point
# =========================================================

# =========================== ! ===========================
# Do not change the class name, attributes name or method name !
# =========================================================
class MathDescription:
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
            'R_core': 5,
            'thickness': 5,
            'sld_value': 1,
        }
        self.coord = 'sph'  # 'car' or 'sph' or 'cyl'

    def get_bound(self) -> tuple[tuple|list, tuple|list]:
        '''re-generate boundary for every method call
        in case that params are altered in software.
        return coordinates in Cartesian coordinates.
        '''
        bound_max = (self.params['R_core']+self.params['thickness']) * torch.ones(3)
        bound_min = -bound_max
        return bound_min.tolist(), bound_max.tolist()

    def sld(self, u: Tensor, v: Tensor, w: Tensor) -> Tensor:
        ''' calculate sld values of certain coordinates
        Args:
            u, v, w: coordinates in self.coord
                x, y, z if self.coord = 'cat'
                r, theta, phi if self.coord = 'sph'
                rho, theta, z if self.coord = 'cyl'
        '''
        device = u.device
        r, theta, phi = u, v, w
        R = self.params['R_core']
        t = self.params['thickness']
        sld_value = self.params['sld_value']
        
        in_model_index = torch.zeros_like(r).to(device)
        in_model_index[(r>=R) & (r<=(R+t))] = 1.
        # sld = r * torch.cos(theta) * torch.sin(phi) * in_model_index
        sld = sld_value * in_model_index
        return sld