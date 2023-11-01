"""
A template of hollow sphere math model
with various sld equal to the x coordinate of certain point

# ! Do not change the class name, attributes name or method name !
"""

from typing import Any, Literal
import torch
from torch import Tensor


class MathModelClass:
    '''to generate a 3D model from a mathematical description
    for example: a spherical shell is "x**2+y**2+z**2 >= R_core**2 and x**2+y**2+z**2 <= (R_core+thickness)**2
    also, in spherical coordinates, a hollow sphere is (r >= R_core) and (r <= R_core+thickness)

    coord:
    - 'car' |in (x, y, z)
    - 'sph' |in (r, theta, phi) |theta: 0~2pi ; phi: 0~pi
    - 'cyl' |in (rho, theta, z) |theta:0-2pi
    '''
    def __init__(self) -> None:
        """must at least have these 2 attributes:
        self.params: dict
        self.coord: Literal['car', 'sph', 'cyl']
        """
        self.params: dict[str, Any] = {
            'R': 20,
            'H': 100,
            'sld_value': 1,
        }
        self.coord: Literal['car', 'sph', 'cyl'] = 'cyl'

    def get_bound(self) -> tuple[tuple|list, tuple|list]:
        """re-generate boundary for every method call
        in case that params are altered in software.
        return coordinates in Cartesian coordinates.

        Returns:
            tuple[tuple|list, tuple|list]: min and max points
        """
        bound_max = (self.params['R']) * torch.ones(3)
        bound_max[-1] = self.params['H']/2
        bound_min = -bound_max
        return bound_min.tolist(), bound_max.tolist()

    def sld(self, u: Tensor, v: Tensor, w: Tensor) -> Tensor:
        """Calculate sld values of certain coordinates.
        u, v, w means:
        x, y, z if self.coord=='car';
        r, theta, phi if self.coord=='sph';
        rho, theta, z if self.coord=='cyl';

        Args:
            u (Tensor): 1st coord
            v (Tensor): 2nd coord
            w (Tensor): 3rd coord

        Returns:
            Tensor: sld values of each coordinates
        """
        device = u.device
        rho, theta, z = u, v, w
        R = self.params['R']
        H = self.params['H']
        sld_value = self.params['sld_value']
        
        in_model_index = torch.zeros_like(rho).to(device)
        in_model_index[(rho<=R)&(torch.abs(z)<=H/2)] = 1.
        sld = sld_value * in_model_index
        return sld