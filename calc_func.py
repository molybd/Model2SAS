'''Compute-intensive functions in model2sas.
Use a seperated module to use pytorch with cuda support more flexibly.
'''

import time

import tqdm
import numpy as np
from scipy.special import sph_harm, spherical_jn

from utility import convert_coord

global TORCH_AVAILABLE
global CUDA_AVAILABLE
global USE_TORCH
global DEVICE

USE_TORCH = False
DEVICE = 'cpu'

try:
    import torch
    #torch = __import__('torch')
    TORCH_AVAILABLE = True
except:
    TORCH_AVAILABLE = False

if TORCH_AVAILABLE:
    CUDA_AVAILABLE = torch.cuda.is_available()
else:
    CUDA_AVAILABLE = False


if TORCH_AVAILABLE:
    check1 = '√'
    if CUDA_AVAILABLE:
        check2 = '√'
    else:
        check2 = 'X'
else:
    check1, check2 = 'X', 'X'
print('-------------------')
print('|  PyTorch  |  {}  |'.format(check1))
print('------------+------')
print('|   CUDA    |  {}  |'.format(check2))
print('-------------------')




def print_time(last_timestamp:float, item:str) -> float:
    '''print time used from last timestamp
    use timestamp = time.time() for the 1st timestamp
    '''
    now = time.time()
    print('{:>20} {:^10}'.format(item, round(now-last_timestamp, 5)))
    timestamp = time.time()
    return timestamp

def moller_trumbore_intersect_count(origins:np.ndarray, ray:np.ndarray, triangles:np.ndarray) -> np.ndarray:
    '''Calculate all the points intersect with 1 triangle
    using Möller-Trumbore intersection algorithm
    see paper https://doi.org/10.1080/10867651.1997.10487468

    Args:
        origins: ndarray, shape == (n, 3)
        ray: ndarray, shape==(3,), direction of ray
        triangles: ndarray, shape==(m,3,3) vertices of a triangle

    Returns:
        ndarray, shape == (n,), 与输入的点(origins)一一对应, 分别为相应点与所有三角形相交的次数
    '''
    timestamp = time.time()
    n = origins.shape[0]
    origins = origins.astype(np.float32)
    ray = ray.astype(np.float32)
    triangles = triangles.astype(np.float32)
    intersect_count = np.zeros(n, dtype=np.float32)
    for triangle in tqdm.tqdm(triangles):
        O = origins
        D = ray
        V0 = triangle[0]
        V1 = triangle[1]
        V2 = triangle[2]
        E1 = V1 - V0
        E2 = V2 - V0
        T = O - V0
        P = np.cross(D, E2)
        Q = np.cross(T, E1)
        det = np.dot(P, E1)
        intersect = np.zeros(n, dtype=np.float32)
        #if abs(det) > np.finfo(np.float32).eps:  # almost impossible
        t, u, v = np.dot(Q,E2)/det, np.dot(T,P)/det, np.dot(Q,D)/det
        intersect[(t>0) & (u>0) & (v>0) & ((u+v)<1)] = 1
        intersect_count += intersect
    timestamp = print_time(timestamp, 'intersection')
    return intersect_count

def sampling_points(s:np.array, orientation_average_offset:int) -> tuple:
    '''generate sampling points by fibonacci grid
    '''
    if TORCH_AVAILABLE and USE_TORCH:
        pass
    else:
        n_on_sphere = s**2
        n_on_sphere = np.rint(n_on_sphere/n_on_sphere[0]) + orientation_average_offset
        n_on_sphere = n_on_sphere.astype(np.int64)
        phi = (np.sqrt(5)-1)/2
        l_n, l_z, l_R = [], [], []
        for R, N in zip(s, n_on_sphere):
            n = np.arange(N)+1
            z = (2*n-1)/N - 1
            R = R*np.ones(N)
            l_n.append(n)
            l_z.append(z)
            l_R.append(R)
        n, z, R = np.concatenate(l_n), np.concatenate(l_z), np.concatenate(l_R)
        x = R*np.sqrt(1-z**2)*np.cos(2*np.pi*n*phi)
        y = R*np.sqrt(1-z**2)*np.sin(2*np.pi*n*phi)
        z = R*z
    return x, y, z

def fibo_grid(N:int, R:float) -> np.ndarray:
    '''generate fibonacci grid
    '''
    phi = (np.sqrt(5)-1)/2
    n = np.arange(N)+1
    z = (2*n-1)/N - 1
    x = np.sqrt(1-z**2)*np.cos(2*np.pi*n*phi)
    y = np.sqrt(1-z**2)*np.sin(2*np.pi*n*phi)
    return R*np.stack((x, y, z), axis=1)

def trilinear_interp(px:np.ndarray, py:np.ndarray, pz:np.ndarray, c:np.ndarray, d:float, x:np.ndarray, y:np.ndarray, z:np.ndarray) -> np.ndarray:
    '''对均匀正方体网格进行插值
    px, py, pz: 1d, 三个坐标序列, size相同
    c: 每个坐标点的值, shape=(px.size, py.size, pz.size)
    d: float 网格间距, 所有格点都是等间距的
    x, y, z: 需要插值点的三个坐标序列
    '''
    ix, iy, iz = (x-px[0])/d, (y-py[0])/d, (z-pz[0])/d
    if TORCH_AVAILABLE and USE_TORCH:
        ix, iy, iz = ix.to(torch.int64), iy.to(torch.int64), iz.to(torch.int64)
    else:
        ix, iy, iz = ix.astype(np.int64), iy.astype(np.int64), iz.astype(np.int64)

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

def sas_fft(grid_sld:np.ndarray, interval:float, q:np.ndarray, n_s:int=400, orientation_average_offset:int=100):
    '''calculate SAS curve by FFT method
    '''
    s = q/(2*np.pi)  # since q=2pi/d, so use s=1/d in fft method

    # determine the actual s to calculate
    # larger n_s gives better result, but comsume more computing power and RAM
    n_l = grid_sld.shape[0]
    n_s = max(n_s, n_l) # n_s must >= n_l
    smin = 1/(n_s*interval)
    smax = 1/interval * (0.5-1/n_s)
    s = s[np.where(s>=smin)]
    s = s[np.where(s<=smax)]

    # fft
    timestamp = time.time()
    if TORCH_AVAILABLE and USE_TORCH:
        grid_sld = torch.from_numpy(grid_sld).to(torch.float32).to(DEVICE)
        F = torch.fft.rfftn(grid_sld, s=(n_s, n_s, n_s))
        F = torch.fft.fftshift(F, dim=(0,1))
        I_grid = torch.real(F)**2 + torch.imag(F)**2
    else:
        F = np.fft.rfftn(grid_sld, (n_s, n_s, n_s))
        F = np.fft.fftshift(F, axes=(0,1))
        I_grid = np.real(F)**2 + np.imag(F)**2  # faster than abs(F)**2
    del F
    timestamp = print_time(timestamp, 'fft')

    # generate coordinates to interpolate
    # 每一个q值对应的球面取多少个取向进行平均
    n_on_sphere = s**2
    n_on_sphere = np.rint(n_on_sphere/n_on_sphere[0]) + orientation_average_offset
    l = []
    for R, N in zip(s, n_on_sphere):
        l.append(fibo_grid(N, R))
    points_to_interpolate = np.vstack(l)
    timestamp = print_time(timestamp, 'fibo grid')

    #### interpolate
    # 因为用的rfft，只有z>=0那一半，因此要将z<0的坐标转换为中心对称的坐标
    points_to_interpolate = np.expand_dims(np.sign(np.sign(points_to_interpolate[:,2])+0.5), axis=1)*points_to_interpolate 
    #I_interp = interpn((s1d, s1d, s1dz), I_grid, points_to_interpolate)
    # 用我自己写的 trilinear_interp 比用 scipy.interpolate.interpn 快
    if TORCH_AVAILABLE and USE_TORCH:
        s1d = torch.fft.fftfreq(n_s, d=interval).to(DEVICE)
        s1d = torch.fft.fftshift(s1d)
        s1dz = torch.fft.rfftfreq(n_s, d=interval).to(DEVICE)
        ds = s1d[1] - s1d[0]
        points_to_interpolate = torch.from_numpy(points_to_interpolate).to(torch.float32).to(DEVICE)
        I_interp = trilinear_interp(s1d, s1d, s1dz, I_grid, ds, points_to_interpolate[:,0], points_to_interpolate[:,1], points_to_interpolate[:,2])
        I_interp = I_interp.cpu().numpy()
    else:
        s1d = np.fft.fftfreq(n_s, d=interval)
        s1d = np.fft.fftshift(s1d)
        s1dz = np.fft.rfftfreq(n_s, d=interval)
        ds = s1d[1] - s1d[0]
        I_interp = trilinear_interp(s1d, s1d, s1dz, I_grid, ds, points_to_interpolate[:,0], points_to_interpolate[:,1], points_to_interpolate[:,2])
    del I_grid
    timestamp = print_time(timestamp, 'interpolate')

    # orientation average
    I = []
    begin_index = 0
    for N in n_on_sphere:
        N = int(N)
        Ii = np.average(I_interp[begin_index:begin_index+N])
        I.append(Ii)
        begin_index += N
    I = np.array(I)
    timestamp = print_time(timestamp, 'average')

    q = 2*np.pi*s
    return q, I


def sas_debyefunc(x:np.ndarray, y:np.ndarray, z:np.ndarray, sld:np.ndarray, q:np.ndarray) -> tuple:
    '''calculate SAS curve by debye function
    '''
    timestamp = time.time()
    q, x, y, z, sld = q.astype('float32'), x.astype('float32'), y.astype('float32'), z.astype('float32'), sld.astype('float32')
    dx = x.reshape((x.size,1)) - x.reshape((1,x.size))
    dy = y.reshape((y.size,1)) - y.reshape((1,y.size))
    dz = z.reshape((z.size,1)) - z.reshape((1,x.size))
    d = np.sqrt(dx**2 + dy**2 + dz**2)

    if TORCH_AVAILABLE and USE_TORCH:
        sld = torch.from_numpy(sld).to(DEVICE)
        d = torch.from_numpy(d).to(DEVICE)
        l = []
        for qi in tqdm.tqdm(q):
            qd = qi*d
            fourier_core = torch.sin(qd)/qd
            fourier_core[d<=1e-11] = 1
            Ii = torch.einsum('i,j,ij', sld, sld, fourier_core)
            l.append(Ii)
        I = torch.stack(l).cpu().numpy()
    else:
        l = []
        for qi in tqdm.tqdm(q):
            qd = qi*d
            fourier_core = np.sin(qd)/qd
            fourier_core[d<=1e-11] = 1
            Ii = np.einsum('i,j,ij', sld, sld, fourier_core)
            l.append(Ii)
        I = np.array(l)
    timestamp = print_time(timestamp, 'debye func')
    return q, I

def sas_sphharm(x:np.ndarray, y:np.ndarray, z:np.ndarray, sld:np.ndarray, q:np.ndarray, lmax:int=50) -> tuple:
    '''calculate SAS curve by spherical harmonics
    '''
    lmax = int(lmax)
    r, theta, phi = convert_coord(x, y, z, 'car', 'sph')
    l = np.linspace(0, lmax, num=lmax+1, endpoint=True, dtype='int16')  # (l,)
    l_ext, m = [], []
    for li in l:
        for mi in range(-li, li+1):
            l_ext.append(li)
            m.append(mi)
    l_ext = np.array(l_ext, dtype='int16')  # (m,)
    m = np.array(m, dtype='int16')  # (m,)

    n_r, n_l, n_m, n_q = r.size, l.size, m.size, q.size

    timestamp = time.time()

    ##### calculate Ylm #####
    theta_ext2 = np.reshape(theta, (n_r, 1))  # (r,) -> (r, 1)
    phi_ext2 = np.reshape(phi, (n_r, 1))  # (r,) -> (r, 1)
    l_ext2 = np.reshape(l_ext, (1, n_m))  # (m,) -> (1, m)
    m_ext2 = np.reshape(m, (1, n_m))  # (m,) -> (1, m)
    #timestamp = print_time(timestamp, 'Ylm preparation')
    Ylm = sph_harm(m_ext2, l_ext2, theta_ext2, phi_ext2).astype(np.complex64)  # broadcast to (r, m) float64
    del theta_ext2, phi_ext2, l_ext2, m_ext2
    timestamp = print_time(timestamp, 'Ylm')
    #########################

    ##### calculate Alm0 without jl #####
    il = complex(0,1)**l  # (l,)
    il_list = []
    for i in range(n_l):
        il_list += [il[i]]*(2*i+1)
    il = np.array(il_list, dtype='complex64')  # (m,)
    #timestamp = print_time(timestamp, 'il')

    Alm0_without_jl = np.einsum('m,r,rm->rm', il, sld, Ylm)
    #timestamp = print_time(timestamp, 'Alm0')
    del il, Ylm
    ############################

    rq = np.einsum('r,q->rq', r, q)  #(r, q)
    rq = np.reshape(rq, (n_r, 1, n_q))  # (r, 1, q)
    l_ext1 = np.reshape(l, (1, n_l, 1))  # (1, l, 1)
    jl_ext1 = spherical_jn(l_ext1, rq)  # broadcast to (r, l, q)
    timestamp = print_time(timestamp, 'jl')
    # (r, l, q) -> (r, m, q)
    jl_ext1 = jl_ext1.astype(np.float32)

    jl = []
    for j in range(n_l):
        jl += [jl_ext1[:,j,:]]*(2*j+1)
    jl = np.array(jl, dtype=np.float32)
    jl = np.swapaxes(jl, 0, 1)  # (r, m, q)
    del rq, jl_ext1
    timestamp = print_time(timestamp, 'jl_rlq->rmq')

    Alm0_without_jl_real, Alm0_without_jl_imag = np.real(Alm0_without_jl), np.imag(Alm0_without_jl)
    Alm_real = np.einsum('rm,rmq->mq', Alm0_without_jl_real, jl)
    Alm_imag = np.einsum('rm,rmq->mq', Alm0_without_jl_imag, jl)
    #Alm = Alm_real + 1j*Alm_imag
    timestamp = print_time(timestamp, 'Alm')

    I = 16 * np.pi**2 * np.sum(Alm_real**2+Alm_imag**2, axis=0) #(q,)
    del jl, Alm_real, Alm_imag, Alm0_without_jl_real, Alm0_without_jl_imag
    return q, I