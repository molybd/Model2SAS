'''Simulation of a 2d detector.
'''

import torch
from torch import Tensor

class Detector:
    '''Simulation of a 2d detector.
    In a coordinate system where sample position as origin,
    beam direction as positive Y axis.
    All length unit should be meter except wavelength.
    Output q unit will be reverse wavelength unit.
    '''
    def __init__(self, resolution: tuple[int, int], pixel_size: float) -> None:
        x = torch.arange(resolution[0], dtype=torch.float32)
        z = torch.arange(resolution[1], dtype=torch.float32)
        x, z = pixel_size*x, pixel_size*z
        cx, cz = (x[0]+x[-1])/2, (z[0]+z[-1])/2
        x, z = x - cx, z - cz
        x, z = torch.meshgrid(x, z, indexing='ij')
        y = torch.zeros_like(x, dtype=torch.float32)
        self.x, self.y, self.z = x, y, z
        self.pitch_axis = torch.tensor((1,0,0), dtype=torch.float32)
        self.yaw_axis = torch.tensor((0,0,1), dtype=torch.float32)
        self.roll_axis = torch.tensor((0,1,0), dtype=torch.float32)
        self.sdd = 0.
        self.resolution = resolution
        self.pixel_size = pixel_size

    def get_center(self) -> Tensor:
        cx = (self.x[0,0] + self.x[-1,-1]) / 2
        cy = (self.y[0,0] + self.y[-1,-1]) / 2
        cz = (self.z[0,0] + self.z[-1,-1]) / 2
        return torch.tensor((cx, cy, cz))

    def set_sdd(self, sdd: float) -> None:
        delta_sdd = sdd - self.sdd
        self.y = self.y + delta_sdd
        self.sdd = sdd

    def translate(self, vx: float, vz: float, vy: float = 0.) -> None:
        self.x = self.x + vx
        self.z = self.z + vz
        self.y = self.y + vy
        self.sdd = self.sdd + vy

    def _euler_rodrigues_rotate(self, coord: Tensor, axis: Tensor, angle: float) -> Tensor:
        '''Rotate coordinates by euler rodrigues rotate formula.
        coord.shape = (n,3)
        '''
        ax = axis / torch.sqrt(torch.sum(axis**2))
        ang = torch.tensor(angle)
        a = torch.cos(ang/2)
        w = ax * torch.sin(ang/2)

        x = coord
        wx = -torch.linalg.cross(x, w.expand_as(x), dim=-1)
        x_rotated = x + 2*a*wx + 2*(-torch.linalg.cross(wx, w.expand_as(wx), dim=-1))
        return x_rotated

    def _rotate(self, rotation_type: str, angle: float) -> None:
        if rotation_type == 'pitch':
            axis = self.pitch_axis
            self.yaw_axis = self._euler_rodrigues_rotate(self.yaw_axis, axis, angle)
            self.roll_axis = self._euler_rodrigues_rotate(self.roll_axis, axis, angle)
        elif rotation_type == 'yaw':
            axis = self.yaw_axis
            self.pitch_axis = self._euler_rodrigues_rotate(self.pitch_axis, axis, angle)
            self.roll_axis = self._euler_rodrigues_rotate(self.roll_axis, axis, angle)
        elif rotation_type == 'roll':
            axis = self.roll_axis
            self.pitch_axis = self._euler_rodrigues_rotate(self.pitch_axis, axis, angle)
            self.yaw_axis = self._euler_rodrigues_rotate(self.yaw_axis, axis, angle)
        center = self.get_center()

        x1d, y1d, z1d = self.x.flatten(), self.y.flatten(), self.z.flatten()
        coord = torch.stack((x1d, y1d, z1d), dim=1)
        coord = coord - center
        rotated_coord = self._euler_rodrigues_rotate(
            coord, axis, angle
        )
        rotated_coord = rotated_coord + center
        x, y, z = torch.unbind(rotated_coord, dim=-1)
        self.x, self.y, self.z = x.reshape(self.resolution), y.reshape(self.resolution), z.reshape(self.resolution)        

    def pitch(self, angle: float) -> None:
        self._rotate('pitch', angle)

    def yaw(self, angle: float) -> None:
        self._rotate('yaw', angle)

    def roll(self, angle: float) -> None:
        self._rotate('roll', angle)

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

    def get_q_range(self, wavelength: float) -> tuple[float, float]:
        qx, qy, qz = self.get_reciprocal_coord(wavelength)
        q = torch.sqrt(qx**2 + qy**2 + qz**2)
        return q.min().item(), q.max().item()

    def get_beamstop_mask(self, d: float) -> Tensor:
        '''pattern must have the same shape as self.x, y, z
        '''
        mask = torch.ones(self.resolution, dtype=torch.float32)
        mask[(self.x**2+self.z**2) <= (d/2)**2] = 0.
        return mask
        