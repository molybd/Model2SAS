'''Compute-intensive functions in model2sas.
All based on pytorch instead of numpy.
'''

# import numpy as np
import torch
from torch import Tensor
import taichi as ti
import taichi.math as tm

from utils import timer

ti.init(ti.gpu)

# * rewritten with taichi for better speed
# * same api remains
# @timer(level=2)
def _torch_moller_trumbore_intersect_count(origins: Tensor, ray: Tensor, triangles: Tensor) -> Tensor:
    '''Calculate all the points intersect with 1 triangle
    using Möller-Trumbore intersection algorithm
    see paper https://doi.org/10.1080/10867651.1997.10487468

    Args:
        origins: Tensor, size=(n, 3)
        ray: Tensor, size=(3,), direction of ray
        triangles: Tensor, size=(m,3,3) vertices of a triangle

    Returns:
        Tensor, size=(n,), 与输入的点(origins)一一对应, 分别为相应点与所有三角形相交的次数
    '''
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
    '''Calculate all the points intersect with 1 triangle
    using Möller-Trumbore intersection algorithm
    see paper https://doi.org/10.1080/10867651.1997.10487468

    Args:
        origins: Tensor, size=(n, 3)
        ray: Tensor, size=(3,), direction of ray
        triangles: Tensor, size=(m,3,3) vertices of a triangle

    Returns:
        Tensor, size=(n,), 与输入的点(origins)一一对应, 分别为相应点与所有三角形相交的次数
    '''
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
def moller_trumbore_intersect_count(origins: Tensor, ray: Tensor, triangles: Tensor, backend: str = 'torch') -> Tensor:
    '''Calculate all the points intersect with 1 triangle
    using Möller-Trumbore intersection algorithm
    see paper https://doi.org/10.1080/10867651.1997.10487468

    Args:
        origins: Tensor, size=(n, 3)
        ray: Tensor, size=(3,), direction of ray
        triangles: Tensor, size=(m,3,3) vertices of a triangle

    Returns:
        Tensor, size=(n,), 与输入的点(origins)一一对应, 分别为相应点与所有三角形相交的次数
    '''
    if backend == 'taichi':
        intersect_count = _taichi_moller_trumbore_intersect_count(origins, ray, triangles)
    else:
        intersect_count = _torch_moller_trumbore_intersect_count(origins, ray, triangles)
    return intersect_count

@timer(level=2)
def sampling_points(s: Tensor, n_on_sphere: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    ''' Generate sampling points for orientation average
    using fibonacci grid
    s and n_on_sphere should have same shape
    Calculate on cpu for convinience, transfered to designate device
    '''
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
def trilinear_interp(x:Tensor, y:Tensor, z:Tensor, px:Tensor, py:Tensor, pz:Tensor, c:Tensor, d:float | Tensor) -> Tensor:
    '''对等间距网格进行三线性插值
    ATTENTION:
        当网格值c是复数时，根据我这里的实际测试结果，这个函数等效于对实部和虚部分别进行插值
    Parameters:
        x, y, z: 待插值的三个坐标序列，(x[i], y[i], z[i])代表待插值的某点坐标
        px, py, pz: 已知数值的点的三个坐标序列，只有一维，也就是说没有进行meshgrid
        c: 每个坐标点的值, shape=(px.size, py.size, pz.size)
        d: float 网格间距, 所有格点都是等间距的
    '''
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