"""some useful utility functions
"""

from typing import Literal

import time
import sys
import functools

import torch
from torch import Tensor
from loguru import logger
from art import text2art


WELCOME_MESSAGE = str(text2art("Model2SAS")).rstrip('\n') + '''
===== Small-angle scattering simulation from 3d models =====

🏠️ Github: https://github.com/molybd/Model2SAS
📄 Please cite: Li, Mu and Yin, Panchao, Model2SAS: software for small-angle scattering data calculation from custom shapes., J. Appl. Cryst., 2022, 55, 663-668. https://doi.org/10.1107/S1600576722003600
'''

LOG: bool = False
LOG_FORMAT_STR: str = '<cyan>{time:YYYY-MM-DD HH:mm:ss.SSS}</cyan> | <level>{level: <8}</level> | <level>{message}</level>'

func_tier: int = 0

def set_log_state(state: bool) -> None:
    global LOG
    LOG = state
    
def add_logger(sink=sys.stderr, format=LOG_FORMAT_STR, **kwargs):
    logger.add(sink, format=format, **kwargs)
    
logger.remove(0)
logger.add(sys.stderr, format=LOG_FORMAT_STR)

def log(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        global func_tier
        if LOG:
            logger.info(f'[{" ":>11}] {"|  "*func_tier}○ {func.__name__}')
            func_tier += 1
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        time_cost = time.perf_counter() - start_time
        if LOG:
            func_tier -= 1
            logger.success(f'[{time_cost:>9.6f} s] {"|  "*func_tier}● {func.__name__}')
        return result
    return wrapper


class MathModelClassBase:
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
        self.params: dict = {}
        self.coord: Literal['car', 'sph', 'cyl'] = 'car'

    def get_bound(self) -> tuple[tuple|list, tuple|list]:
        """re-generate boundary for every method call
        in case that params are altered in software.
        return coordinates in Cartesian coordinates.

        Returns:
            tuple[tuple|list, tuple|list]: min and max points
        """
        return (-1*torch.ones(3)).tolist(), torch.ones(3).tolist()

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
        return torch.zeros_like(u)


def gen_math_model_class(
        name: str = 'SpecificMathModelClass',
        params: dict | None = None,
        coord: Literal['car', 'sph', 'cyl'] = 'car',
        bound_point: tuple[str, str, str] = ('1', '1', '1'),
        shape_description: str = ':',
        sld_description: str = 'False',
    ):
    """Generate a math modol class dynamically by description strings.

    Args:
        name (str, optional): name of math model class. Defaults to 'SpecificMathModelClass'.
        params (dict | None, optional): params dict. Defaults to None.
        coord (Literal[&#39;car&#39;, &#39;sph&#39;, &#39;cyl&#39;], optional): coordination type. Defaults to 'car'.
        bound_point (tuple[str, str, str], optional): max point of the box containing the whole model. Can be expressions by params. Defaults to ('1', '1', '1').
        shape_description (_type_, optional): describe shape by coordinates and params. Defaults to ':'.
        sld_description (str, optional): describe sld values by coordinates and params. Defaults to 'False'.

    Returns:
        MathModelClassBase: math model class
    """
    if params is None:
        params = {}
        
    def init(self) -> None:
        self.params = params
        self.coord = coord

    def get_bound(self) -> tuple[tuple|list, tuple|list]:
        for key, value in self.params.items():
            exec('{} = {}'.format(key, value))
        bound_max = torch.ones(3)
        bound_max[0] = eval(bound_point[0])
        bound_max[1] = eval(bound_point[1])
        bound_max[2] = eval(bound_point[2])
        bound_min = -1 * bound_max
        return bound_min.tolist(), bound_max.tolist()
    
    def sld(self, u: Tensor, v: Tensor, w: Tensor) -> Tensor:
        device = u.device
        for key, value in self.params.items():
            exec('{} = {}'.format(key, value))
        shape_index = torch.zeros_like(u, device=device)
        exec('shape_index[{}] = 1.0'.format(shape_description))
        sld = torch.ones_like(u, device=device)
        sld = eval('({}) * sld'.format(sld_description))
        sld = shape_index * sld
        return sld

    attr = dict(
        __init__ = init,
        get_bound = get_bound,
        sld = sld,
        info = dict(
            params = params,
            coord = coord, 
            bound_point = bound_point,
            shape_description = shape_description,
            sld_description = sld_description,
        )
    )
    math_model_class = type(name, (MathModelClassBase,), attr)
    return math_model_class


def gen_math_model_class_sourcecode(
        params: dict,
        coord: Literal['car', 'sph', 'cyl'], 
        bound_point: tuple[str, str, str],
        shape_description: str,
        sld_description: str,
        ) -> str:
    """Generate source code string of math model class for file saving.
    Args mainly get from MathModelClass.info dict.

    Args:
        params (dict): _description_
        coord (Literal[&#39;car&#39;, &#39;sph&#39;, &#39;cyl&#39;]): _description_
        bound_point (tuple[str, str, str]): _description_
        shape_description (str): _description_
        sld_description (str): _description_

    Returns:
        str: source code string of MathModelClass
    """

    source_code = """
import torch
from torch import Tensor

class MathModelClass:
    
    def __init__(self) -> None:
        self.params = {{
{}
        }}
        self.coord = '{}'
    """.format(
        '\n'.join(['{}"{}": {},'.format(' '*12, str(key), str(value)) for key, value in params.items()]),
        coord,
    )

    source_code += """
    def get_bound(self) -> tuple[tuple|list, tuple|list]:
{}
        bound_max = torch.tensor(({}, {}, {}), dtype=torch.float32)
        bound_min = -1 * bound_max
        return bound_min.tolist(), bound_max.tolist()
    """.format(
        '\n'.join(['{}{} = self.params["{}"]'.format(' '*8, str(key), str(key)) for key in params.keys()]),
        *bound_point,
    )

    source_code += """
    def sld(self, u: Tensor, v: Tensor, w: Tensor) -> Tensor:
        device = u.device
{}
        shape_index = torch.zeros_like(u, device=device)
        shape_index[{}] = 1.0
        sld = torch.ones_like(u, device=device)
        sld = ({}) * sld
        sld = shape_index * sld
        return sld
    """.format(
        '\n'.join(['{}{} = self.params["{}"]'.format(' '*8, str(key), str(key)) for key in params.keys()]),
        shape_description,
        sld_description,
    )
    return source_code


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

    def _rotate(self, rotation_type: Literal['pitch', 'yaw', 'roll'], angle: float) -> None:
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
        else:
            raise ValueError('Unsupported rotation type: {}'.format(rotation_type))
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


def save_pdb_file(filename: str, x: Tensor, y: Tensor, z:Tensor, sld: Tensor, atom_name: str = 'CA', temperature_factor: float = 0.0, element_symbol: str = 'C') -> None:
    """Convert a lattice model to a pdb file
    for calculation by other software like CRYSOL.
    Only preserve sld!=0 points with uniform sld value.

    Args:
        x (Tensor): x coordinates
        y (Tensor): x coordinates
        z (Tensor): x coordinates
        sld (Tensor): sld values at each coordinates
        filename (str): output pdb file name
    """    
    sld = sld.flatten()
    x = x.flatten()[sld!=0.]
    y = y.flatten()[sld!=0.]
    z = z.flatten()[sld!=0.]
    sld = sld[sld!=0]
    
    lines = ['REMARK 265 EXPERIMENT TYPE: THEORETICAL MODELLING\n']
    for i, (xi, yi, zi) in enumerate(zip(x, y, z)):
        lines.append(
            f'{"ATOM":<6}{i+1:>5d} {atom_name:<4} {"ASP":>3} {"A":>1}{1:>4d}    {xi:>8.3f}{yi:>8.3f}{zi:>8.3f}{1.0:>6.2f}{temperature_factor:>6.2f}          {element_symbol:>2}  \n'
        )
    with open(filename, 'w', encoding='utf-8') as f:
        f.writelines(lines)