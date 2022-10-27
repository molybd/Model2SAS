'''Compute-intensive functions in model2sas.
Use a seperated module to use pytorch with cuda support more flexibly.
'''

# import numpy as np
import torch
from torch import Tensor

from utility_new import timer


@timer
def moller_trumbore_intersect_count(origins: Tensor, ray: Tensor, triangles: Tensor) -> Tensor:
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
    n = origins.size()[0]
    intersect_count = torch.zeros(n, dtype=torch.int32)
    E1 = triangles[:,1,:] - triangles[:,0,:]
    E2 = triangles[:,2,:] - triangles[:,0,:]
    for i in range(triangles.size()[0]):
        T = origins - triangles[i,0,:]
        P = torch.linalg.cross(ray, E2[i,:], dim=-1)
        Q = torch.linalg.cross(T, E1[i,:], dim=-1)
        det = torch.dot(P, E1[i,:])
        intersect = torch.zeros(n, dtype=torch.int32)
        t, u, v = torch.matmul(Q,E2[i,:])/det, torch.matmul(T,P)/det, torch.matmul(Q,ray)/det
        intersect[(t>0) & (u>0) & (v>0) & ((u+v)<1)] = 1  # faster than below
        # intersect = torch.where((t>0)&(u>0)&(v>0)&((u+v)<1), 1, 0)
        intersect_count += intersect
    return intersect_count

@timer
def sampling_points(s: Tensor, n_on_sphere: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    ''' generate sampling points for orientation average
    using fibonacci grid
    s and n_on_sphere should have same shape
    '''
    phi = (torch.sqrt(torch.tensor(5))-1)/2
    l_n, l_z, l_R = [], [], []
    for R, N in zip(s, n_on_sphere):
        n = torch.arange(N.int(), dtype=torch.float32)+1
        z = (2*n-1)/N - 1
        R = R*torch.ones(N.int(), dtype=torch.float32)
        l_n.append(n)
        l_z.append(z)
        l_R.append(R)
    n, z, R = torch.cat(l_n), torch.cat(l_z), torch.cat(l_R)
    x = R*torch.sqrt(1-z**2)*torch.cos(2*torch.pi*n*phi)
    y = R*torch.sqrt(1-z**2)*torch.sin(2*torch.pi*n*phi)
    z = R*z
    return x, y, z

@timer
def trilinear_interp(px:Tensor, py:Tensor, pz:Tensor, c:Tensor, d:float, x:Tensor, y:Tensor, z:Tensor) -> Tensor:
    '''对均匀正方体网格进行插值
    px, py, pz: 1d, 三个坐标序列, size相同
    c: 每个坐标点的值, shape=(px.size, py.size, pz.size)
    d: float 网格间距, 所有格点都是等间距的
    x, y, z: 需要插值点的三个坐标序列
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