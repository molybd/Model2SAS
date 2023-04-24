'''some useful utility functions
'''

from typing import Literal

import time
import functools

import torch
from torch import Tensor


def timer(level: int = 0):
    def timer_decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            time_cost = time.time() - start_time
            if level ==0:
                prefix = '✅ '
            else:
                prefix = '⏳' + ' ⬇'*level + ' '
            # print('{} [{:>9}s] {}'.format(prefix, round(time_cost, 5), func.__name__))
            print('[{} {:>2}.{:<5.0f} s] {}{}'.format(
                '⏱',
                int(time_cost),
                round((time_cost-int(time_cost))*1e5, 5),
                prefix,
                func.__name__
                ))
            return result
        return wrapper
    return timer_decorator

def convert_coord(u:Tensor, v:Tensor, w:Tensor, original_coord: Literal['car', 'sph', 'cyl'], target_coord: Literal['car', 'sph', 'cyl']) -> tuple[Tensor, Tensor, Tensor]:
    ''' Convert coordinates
    car: Cartesian coordinates, in (x, y, z)
    sph: spherical coordinates, in (r, theta, phi) | theta: 0~2pi ; phi: 0~pi
    cyl: cylindrical coordinates, in (rho, phi, z) | theta:0-2pi

    Attributes:
        u: 1st coord
        v: 2nd coord
        w: 3rd coord
        original_coord: 'car' or 'sph' or 'cyl'
        target_coord: 'car' or 'sph' or 'cyl'
    
    Return:
        converted u, v, w
    '''
    def car2sph(x:Tensor, y:Tensor, z:Tensor) -> tuple[Tensor, Tensor, Tensor]:
        '''convert cartesian coordinates to spherical coordinates
        '''
        r = torch.sqrt(x**2 + y**2 + z**2)
        phi = torch.arccos(z/r) # when r=0, output phi=nan
        phi = torch.nan_to_num(phi, nan=0.) # convert nan to 0
        theta = torch.arctan2(y, x) # range [-pi, pi]
        theta = torch.where(theta<0, theta+2*torch.pi, theta) # convert range to [0, 2pi]
        return r, theta, phi
    def car2cyl(x:Tensor, y:Tensor, z:Tensor) -> tuple[Tensor, Tensor, Tensor]:
        '''convert cartesian coordinates to cylindrical coordinates
        '''
        rho = torch.sqrt(x**2+y**2)
        theta = torch.arctan2(y, x) # range [-pi, pi]
        theta = theta + (1-torch.sign(torch.sign(theta)+1))*2*torch.pi # convert range to [0, 2pi]
        return rho, theta, z
    def sph2car(r:Tensor, theta:Tensor, phi:Tensor) -> tuple[Tensor, Tensor, Tensor]:
        '''convert spherical coordinates to cartesian coordinates
        '''
        x = r * torch.cos(theta) * torch.sin(phi)
        y = r * torch.sin(theta) * torch.sin(phi)
        z = r * torch.cos(phi)
        return x, y, z
    def cyl2car(rho:Tensor, theta:Tensor, z:Tensor) -> tuple[Tensor, Tensor, Tensor]:
        '''convert cylindrical coordinates to cartesian coordinates
        '''
        x = rho * torch.cos(theta)
        y = rho * torch.sin(theta)
        return x, y, z

    # all convert to Cartesian coordinates
    if original_coord == 'sph':
        x, y, z = sph2car(u, v, w)
    elif original_coord == 'cyl':
        x, y, z = cyl2car(u, v, w)
    elif original_coord == 'car':
        x, y, z = u, v, w
    else:
        raise ValueError('Unsupported coordinates: {}'.format(original_coord))
    
    # then convert to desired coordinates
    if target_coord == 'sph':
        return car2sph(x, y, z)
    elif target_coord == 'cyl':
        return car2cyl(x, y, z)
    elif target_coord == 'car':
        return x, y, z
    else:
        raise ValueError('Unsupported coordinates: {}'.format(target_coord))

def abi2modarg(t: Tensor) -> tuple[Tensor, Tensor]:
    '''Change a complex tensor from a+bi expression
    to mod*exp(i*arg) expression.
    '''
    mod = torch.sqrt(t.real**2 + t.imag**2)
    arg = torch.arctan2(t.imag, t.real)
    return mod, arg

def modarg2abi(mod: Tensor, arg: Tensor) -> Tensor:
    '''Change a complex tensor from mod*exp(i*arg)
    expression to a+bi expression.
    '''
    return mod * torch.complex(torch.cos(arg), torch.sin(arg))


class MathModelClassBase:
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
        self.params: dict = {}
        self.coord: Literal['car', 'sph', 'cyl'] = 'car'  # 'car' or 'sph' or 'cyl'

    def get_bound(self) -> tuple[tuple|list, tuple|list]:
        '''re-generate boundary for every method call
        in case that params are altered in software.
        return coordinates in Cartesian coordinates.
        '''
        return (-1*torch.ones(3)).tolist(), torch.ones(3).tolist()

    def sld(self, u: Tensor, v: Tensor, w: Tensor) -> Tensor:
        ''' calculate sld values of certain coordinates
        Args:
            u, v, w: coordinates in self.coord
                x, y, z if self.coord = 'car'
                r, theta, phi if self.coord = 'sph'
                rho, theta, z if self.coord = 'cyl'
        '''
        return torch.zeros_like(u)


def gen_math_model_class(
        name: str = 'SpecificMathModelClass',
        params: dict | None = None,
        coord: Literal['car', 'sph', 'cyl'] = 'car',
        bound_point: tuple[str, str, str] = ('1', '1', '1'),
        shape_description: str = ':',
        sld_description: str = 'False',
    ):
    '''Generate a math modol class dynamically.
    '''
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
    '''Generate source code string of math model class
    for file saving.
    '''

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
