'''some useful utility functions
'''
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
                prefix = '✔'
            else:
                prefix = '⬇'
            print('{} [{:>9}s] {}'.format(prefix, round(time_cost, 5), func.__name__))
            return result
        return wrapper
    return timer_decorator

def convert_coord(u:Tensor, v:Tensor, w:Tensor, original_coord:str, target_coord:str) -> tuple[Tensor, Tensor, Tensor]:
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
    else:
        x, y, z = u, v, w
    
    # then convert to desired coordinates
    if target_coord == 'car':
        return x, y, z
    elif target_coord == 'sph':
        return car2sph(x, y, z)
    elif target_coord == 'cyl':
        return car2cyl(x, y, z)

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