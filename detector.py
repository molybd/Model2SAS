'''Simulation of a 2d detector.
'''

import torch
from torch import Tensor

class Detector:
    '''Simulation of a 2d detector.
    In a coordinate system where sample position as origin,
    beam direction as positive Y axis.
    '''
    def __init__(self, resolution: tuple[float, float], pixel_size: float) -> None:
        x = torch.arange(resolution[0], dtype=torch.float32)
        z = torch.arange(resolution[1], dtype=torch.float32)
        x, z = pixel_size*x, pixel_size*z
        cx, cz = (x[-1]-x[0])/2, (z[-1]-z[0])/2
        x, z = x - cx, z - cz
        x, z = torch.meshgrid(x, z, indexing='ij')
        y = torch.zeros_like(x, dtype=torch.float32)
        self.x, self.y, self.z = x, y, z
        self.sdd: float

    def set_sdd(self, sdd: float) -> None:
        self.y = self.y + sdd
        self.sdd = sdd

    def translate(self, vx: float, vz: float) -> None:
        self.x = self.x + vx
        self.z = self.z + vz

    def _real_coord_to_reciprocal_coord(self, x: Tensor, y: Tensor, z: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        '''In a coordinate system where sample position as origin,
        beam direction as positive Y axis, calculate the corresponding
        reciprocal coordinates (without multiply wave vector
        k=2pi/wavelength) by coordinates (x,y,z) in this space.
        '''
        mod = torch.sqrt(x**2 + y**2 + z**2)
        unit_vector_ks_x, unit_vector_ks_y, unit_vector_ks_z = x/mod, y/mod, z/mod
        unit_vector_ki_x, unit_vector_ki_y, unit_vector_ki_z = 0., 1., 0.
        rx = unit_vector_ks_x - unit_vector_ki_x
        ry = unit_vector_ks_y - unit_vector_ki_y
        rz = unit_vector_ks_z - unit_vector_ki_z
        return rx, ry, rz

    def get_reciprocal_coord(self, wavelength: float) -> tuple[Tensor, Tensor, Tensor]:
        k = 2*torch.pi / wavelength
        rx, ry, rz = self._real_coord_to_reciprocal_coord(self.x, self.y, self.z)
        qx, qy, qz = k*rx, k*ry, k*rz
        return qx, qy, qz
        