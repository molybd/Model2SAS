"""Compute-related functions in model2sas.
All based on pytorch instead of numpy.
"""

from typing import Literal

# import numpy as np
import torch
from torch import Tensor

from .utils import timer

try:
    import taichi as ti
    import taichi.math as tm
    ti.init(ti.gpu)
    TAICHI_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    TAICHI_AVAILABLE = False
    print('❌ taichi not available')


def convert_coord(u:Tensor, v:Tensor, w:Tensor, original_coord: Literal['car', 'sph', 'cyl'], target_coord: Literal['car', 'sph', 'cyl']) -> tuple[Tensor, Tensor, Tensor]:
    """Convert coordinates
    car: Cartesian coordinates, in (x, y, z)
    sph: spherical coordinates, in (r, theta, phi); theta: 0~2pi ; phi: 0~pi
    cyl: cylindrical coordinates, in (rho, theta, z); theta: 0-2pi

    Args:
        u (Tensor): 1st coord
        v (Tensor): 2nd coord
        w (Tensor): 3rd coord
        original_coord (Literal[&#39;car&#39;, &#39;sph&#39;, &#39;cyl&#39;]): original coordinate type to be converted
        target_coord (Literal[&#39;car&#39;, &#39;sph&#39;, &#39;cyl&#39;]): target coordinate type

    Raises:
        ValueError: _description_
        ValueError: _description_

    Returns:
        tuple[Tensor, Tensor, Tensor]: converted coordinates
    """
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
    """Change a complex tensor from a+bi expression
    to mod*exp(i*arg) expression.

    Args:
        t (Tensor): complex tensor, with a+bi expresion

    Returns:
        tuple[Tensor, Tensor]: mod and argument of complex number
    """
    mod = torch.sqrt(t.real**2 + t.imag**2)
    arg = torch.arctan2(t.imag, t.real)
    return mod, arg

def modarg2abi(mod: Tensor, arg: Tensor) -> Tensor:
    """Change a complex tensor from mod*exp(i*arg)
    expression to a+bi expression.

    Args:
        mod (Tensor): mod of complex number
        arg (Tensor): argument of complex number

    Returns:
        Tensor: complex tensor
    """
    return mod * torch.complex(torch.cos(arg), torch.sin(arg))


# * rewritten with taichi for better speed
# * same api remains
# @timer(level=2)
def _torch_moller_trumbore_intersect_count(origins: Tensor, ray: Tensor, triangles: Tensor) -> Tensor:
    """Calculate all the points intersect with 1 triangle
    using Möller-Trumbore intersection algorithm
    see paper https://doi.org/10.1080/10867651.1997.10487468

    Args:
        origins (Tensor): shape=(n, 3), points to be determined
        ray (Tensor): shape=(3,), direction of ray
        triangles (Tensor): shape=(m,3,3) vertices of a triangle

    Returns:
        Tensor: size=(n,), 与输入的点(origins)一一对应, 分别为相应点与所有三角形相交的次数
    """
    device = origins.device
    n = origins.size()[0]
    intersect_count = torch.zeros(n, dtype=torch.int32, device=device)
    E1 = triangles[:,1,:] - triangles[:,0,:]
    E2 = triangles[:,2,:] - triangles[:,0,:]
    for i in range(triangles.size()[0]):
        T = origins - triangles[i,0,:]
        P = torch.linalg.cross(ray, E2[i,:], dim=-1)
        Q = torch.linalg.cross(T, E1[i,:].expand_as(T), dim=-1)
        det = torch.dot(P, E1[i,:])
        intersect = torch.zeros(n, dtype=torch.int32, device=device)
        t, u, v = torch.matmul(Q,E2[i,:])/det, torch.matmul(T,P)/det, torch.matmul(Q,ray)/det
        intersect[(t>0) & (u>0) & (v>0) & ((u+v)<1)] = 1  # faster than below
        # intersect = torch.where((t>0)&(u>0)&(v>0)&((u+v)<1), 1, 0)
        intersect_count += intersect
    return intersect_count

@ti.func
def _one_ray_one_triangle_intersect(
        O: tm.vec3,
        D: tm.vec3,
        V0: tm.vec3,
        V1: tm.vec3,
        V2: tm.vec3
    ) -> ti.int32:
    E1 = V1 - V0
    E2 = V2 - V0
    T = O - V0
    P = tm.cross(D, E2)
    Q = tm.cross(T, E1)
    det = tm.dot(P, E1)
    # tuv = tm.vec3(tm.dot(Q, E2), tm.dot(P, T), tm.dot(Q, D)) / det
    t, u, v = tm.dot(Q, E2)/det, tm.dot(P, T)/det, tm.dot(Q, D)/det
    count: ti.int32 = 0
    if t > 0.0 and u > 0.0 and v > 0.0 and (u+v)<1.0:
        count = 1
    return count

# @timer(level=2)
def _taichi_moller_trumbore_intersect_count(origins: Tensor, ray: Tensor, triangles: Tensor) -> Tensor:
    """Calculate all the points intersect with 1 triangle
    using Möller-Trumbore intersection algorithm
    see paper https://doi.org/10.1080/10867651.1997.10487468

    Args:
        origins (Tensor): shape=(n, 3), points to be determined
        ray (Tensor): shape=(3,), direction of ray
        triangles (Tensor): shape=(m,3,3) vertices of a triangle

    Returns:
        Tensor: size=(n,), 与输入的点(origins)一一对应, 分别为相应点与所有三角形相交的次数
    """
    device = origins.device
    # if device.type == 'cuda':
    #     ti.init(ti.cuda)
    # else:
    #     ti.init(ti.cpu)
    
    n, m = origins.shape[0], triangles.shape[0]
    ori = ti.Vector.field(3, ti.f32, shape=(n,))
    ori.from_torch(origins)
    r = ti.Vector.field(3, ti.f32, shape=(1,))
    r.from_torch(ray.unsqueeze(0))
    tri = ti.Vector.field(3, ti.f32, shape=(m, 3))
    tri.from_torch(triangles)

    count_vec = ti.field(ti.int32, shape=(n,))
    @ti.kernel
    def gen_count_vec():
        N, M = ori.shape[0], tri.shape[0]
        for i, j in ti.ndrange(N, M):
            count_vec[i] += _one_ray_one_triangle_intersect(
                ori[i], r[0], tri[j,0], tri[j,1], tri[j,2]
            )
    gen_count_vec()
    intersect_count = count_vec.to_torch(device=device)
    return intersect_count

@timer(level=2)
def moller_trumbore_intersect_count(origins: Tensor, ray: Tensor, triangles: Tensor, backend: Literal['torch', 'taichi'] = 'torch') -> Tensor:
    """Calculate all the points intersect with 1 triangle
    using Möller-Trumbore intersection algorithm
    see paper https://doi.org/10.1080/10867651.1997.10487468

    Args:
        origins (Tensor): shape=(n, 3), points to be determined
        ray (Tensor): shape=(3,), direction of ray
        triangles (Tensor): shape=(m,3,3) vertices of a triangle
        backend (Literal[&#39;torch&#39;, &#39;taichi&#39;], optional): calculation backend, taichi is faster. Defaults to 'torch'.

    Returns:
        Tensor: size=(n,), 与输入的点(origins)一一对应, 分别为相应点与所有三角形相交的次数
    """
    if backend == 'taichi':
        if TAICHI_AVAILABLE:
            intersect_count = _taichi_moller_trumbore_intersect_count(origins, ray, triangles)
        else:
            raise ValueError('taichi not available')
    elif backend == 'torch':
        intersect_count = _torch_moller_trumbore_intersect_count(origins, ray, triangles)
    else:
        raise ValueError('Unsupported backend: {}'.format(backend))
    return intersect_count

@timer(level=2)
def sampling_points(s: Tensor, n_on_sphere: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Generate sampling points for orientation average
    using fibonacci grid
    s and n_on_sphere should have same shape
    Calculate on cpu for convinience, transfered to designate device

    Args:
        s (Tensor): shape=(n,), list of radius of spherical shells in reciprocal space
        n_on_sphere (Tensor): shape=(n,), number of points on each shell

    Returns:
        tuple[Tensor, Tensor, Tensor]: coordinates in reciprocal space x, y, z
    """
    device = s.device
    s = s.to('cpu')
    n_on_sphere = n_on_sphere.to('cpu')
    phi = (torch.sqrt(torch.tensor(5))-1)/2
    l_n, l_z, l_R = [], [], []
    for R, N in zip(s, n_on_sphere):
        n = torch.arange(N.int().item(), dtype=torch.float32) + 1
        z = (2*n-1)/N - 1
        R = R*torch.ones(N.int().item(), dtype=torch.float32)
        l_n.append(n)
        l_z.append(z)
        l_R.append(R)
    n, z, R = torch.cat(l_n), torch.cat(l_z), torch.cat(l_R)
    x = R*torch.sqrt(1-z**2)*torch.cos(2*torch.pi*n*phi)
    y = R*torch.sqrt(1-z**2)*torch.sin(2*torch.pi*n*phi)
    z = R*z
    return x.to(device), y.to(device), z.to(device)

@timer(level=2)
def nearest_interp(x:Tensor, y:Tensor, z:Tensor, px:Tensor, py:Tensor, pz:Tensor, c:Tensor, d:float | Tensor) -> Tensor:
    """Conduct nearest interpolate on equally spaced meshgrid.
    当网格值c是复数时等效于对实部和虚部分别进行插值

    Args:
        x (Tensor): shape=(n,), x coordinates of points to be interpolated
        y (Tensor): shape=(n,), y coordinates of points to be interpolated
        z (Tensor): shape=(n,), z coordinates of points to be interpolated
        px (Tensor): shape=(m1,), x edge grid of meshgrid with known values
        py (Tensor): shape=(m2,), y edge grid of meshgrid with known values
        pz (Tensor): shape=(m3,), z edge grid of meshgrid with known values
        c (Tensor): shape=(m1, m2, m3), values of each of in meshgrid(px, py, pz)
        d (float | Tensor): spacing of meshgrid(px, py, pz), equally spaced

    Returns:
        Tensor: shape=(n,), interpolated values of (x, y, z)
    """
    ix, iy, iz = (x-px[0]+d/2)/d, (y-py[0]+d/2)/d, (z-pz[0]+d/2)/d
    ix, iy, iz = ix.to(torch.int64), iy.to(torch.int64), iz.to(torch.int64) # tensors used as indices must be long, byte or bool tensors
    c_interp = c[ix, iy, iz]
    return c_interp

@timer(level=2)
def trilinear_interp(x:Tensor, y:Tensor, z:Tensor, px:Tensor, py:Tensor, pz:Tensor, c:Tensor, d:float | Tensor) -> Tensor:
    """Conduct trilinear interpolate on equally spaced meshgrid.
    当网格值c是复数时等效于对实部和虚部分别进行插值

    Args:
        x (Tensor): shape=(n,), x coordinates of points to be interpolated
        y (Tensor): shape=(n,), y coordinates of points to be interpolated
        z (Tensor): shape=(n,), z coordinates of points to be interpolated
        px (Tensor): shape=(m1,), x edge grid of meshgrid with known values
        py (Tensor): shape=(m2,), y edge grid of meshgrid with known values
        pz (Tensor): shape=(m3,), z edge grid of meshgrid with known values
        c (Tensor): shape=(m1, m2, m3), values of each of in meshgrid(px, py, pz)
        d (float | Tensor): spacing of meshgrid(px, py, pz), equally spaced

    Returns:
        Tensor: shape=(n,), interpolated values of (x, y, z)
    """
    ix, iy, iz = (x-px[0])/d, (y-py[0])/d, (z-pz[0])/d
    ix, iy, iz = ix.to(torch.int64), iy.to(torch.int64), iz.to(torch.int64) # tensors used as indices must be long, byte or bool tensors

    x0, y0, z0 = px[ix], py[iy], pz[iz]
    x1, y1, z1 = px[ix+1], py[iy+1], pz[iz+1]
    xd, yd, zd = (x-x0)/(x1-x0), (y-y0)/(y1-y0), (z-z0)/(z1-z0)
    
    c_interp = c[ix, iy, iz]*(1-xd)*(1-yd)*(1-zd)
    c_interp += c[ix+1, iy, iz]*xd*(1-yd)*(1-zd)
    c_interp += c[ix, iy+1, iz]*(1-xd)*yd*(1-zd)
    c_interp += c[ix, iy, iz+1]*(1-xd)*(1-yd)*zd
    c_interp += c[ix+1, iy, iz+1]*xd*(1-yd)*zd
    c_interp += c[ix, iy+1, iz+1]*(1-xd)*yd*zd
    c_interp += c[ix+1, iy+1, iz]*xd*yd*(1-zd)
    c_interp += c[ix+1, iy+1, iz+1]*xd*yd*zd
    return c_interp

@timer(level=2)
def euler_rodrigues_rotate(coord: Tensor, axis_local: tuple[float, float, float], angle_local: float) -> Tensor:
    """Central rotation of coordinates by Euler-Rodrigues formula.
    Refer to https://en.wikipedia.org/wiki/Euler%E2%80%93Rodrigues_formula

    Args:
        coord (Tensor): coordinates, shape=(n,3)
        axis_local (tuple[float, float, float]): rotating axis, vector from (0,0,0)
        angle_local (float): rotation angle in radian

    Returns:
        Tensor: rotated coordinates, shape=(n,3)
    """    
    device = coord.device
    ax = torch.tensor(axis_local, dtype=torch.float32, device=device)
    ax = ax / torch.sqrt(torch.sum(ax**2))
    ang = torch.tensor(angle_local, dtype=torch.float32, device=device)
    a = torch.cos(ang/2)
    b = ax[0]*torch.sin(ang/2)
    c = ax[1]*torch.sin(ang/2)
    d = ax[2]*torch.sin(ang/2)
    w = torch.tensor((b, c, d), device=device)

    x = coord
    wx = -torch.linalg.cross(x, w.expand_as(x), dim=-1)
    x_rotated = x + 2*a*wx + 2*(-torch.linalg.cross(wx, w.expand_as(wx), dim=-1))
    return x_rotated
