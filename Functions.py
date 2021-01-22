# -*- coding: UTF-8 -*-

import numpy as np
from scipy.special import sph_harm, spherical_jn, jv


def intensity(q, points, f, lmax):
    q = q.astype('float32')
    q = q.reshape(q.size)  # (q,)
    points_sph = xyz2sph(points)
    r, theta, phi = points_sph[:,0], points_sph[:,1], points_sph[:,2]
    r, theta, phi = r.reshape(r.size), theta.reshape(theta.size), phi.reshape(phi.size)
    r, theta, phi = r.astype('float32'), theta.astype('float32'), phi.astype('float32')  # (r,)
    f = f.astype('float32')
    f = f.reshape(f.size)   # (r,)

    # _ext means extended

    lmax = int(lmax)
    l = np.linspace(0, lmax, num=lmax+1, endpoint=True, dtype='int16')  # (l,)
    l_ext, m = [], []
    for li in l:
        for mi in range(-li, li+1):
            l_ext.append(li)
            m.append(mi)
    l_ext = np.array(l_ext, dtype='int16')  # (m,)
    m = np.array(m, dtype='int16')  # (m,)

    r_ext1 = np.array([ri*np.ones((l.size, q.size)) for ri in r], dtype='float32')  # (r, l, q)
    q_temp = np.array([q]*l.size, dtype='float32')  # (l, q)
    q_ext1 = np.array([q_temp]*r.size, dtype='float32')  # (r, l, q)
    rq_ext1 = r_ext1 * q_ext1  # (r, l, q)
    l_temp = np.array([li*np.ones_like(q) for li in l], dtype='int16')  # (l, q)
    l_ext1 = np.array([l_temp]*r.size, dtype='int16')  # (r, l, q)
    jl = spherical_jn(l_ext1, rq_ext1)  # (r, l, q)

    theta_ext2 = np.array([theta_i*np.ones_like(m) for theta_i in theta], dtype='float32') # (r, m)
    phi_ext2 = np.array([phi_i*np.ones_like(m) for phi_i in phi], dtype='float32') # (r, m)
    l_ext2 = np.array([l_ext]*r.size, dtype='int16')  # (r, m)
    m_ext2 = np.array([m]*r.size, dtype='int16')  # (r, m)
    Ylm = sph_harm(m_ext2, l_ext2, theta_ext2, phi_ext2)  # (r, m)

    fj = f  # (r,)

    # 接下来把各个部分都扩展成 shape=(r, m, q)
    n_r, n_l, n_m, n_q = r.size, l.size, m.size, q.size

    fj_ext3 = np.array([fj_i*np.ones((m.size, q.size)) for fj_i in fj], dtype='float32')  # (r, m, q)

    jl_ext3 = []
    for i in range(n_r):
        temp = []
        for j in range(n_l):
            temp += [ jl[i,j,:] ]*(2*j+1)  # (m, q)
        jl_ext3.append(temp)
    jl_ext3 = np.array(jl_ext3, dtype='float32')  # (r, m, q)

    Ylm_ext3 = np.zeros((n_r, n_m, n_q), dtype='complex64')
    for i in range(n_r):
        for j in range(n_m):
            for k in range(n_q):
                Ylm_ext3[i,j,k] = Ylm[i,j]
    # Ylm_ext3.shape == (r, m, q)

    Sigma1 = fj_ext3 * jl_ext3 * Ylm_ext3  # (r, m, q)
    Sigma1 = np.sum(Sigma1, axis=0)  # (m, q)

    il = complex(0,1)**l  # (l,)
    il_ext4 = []
    for i in range(n_l):
        il_ext4 += [il[i]]*(2*i+1)
    # il_ext4.shape == (m,)
    il_ext5 = np.array([il_ext4_i*np.ones_like(q) for il_ext4_i in il_ext4], dtype='complex64')  # (m, q)

    Alm = il_ext5 * Sigma1  # (m, q)
    Iq = 16 * np.pi**2 * np.sum(np.absolute(Alm)**2, axis=0)  # (q,)

    return Iq



def xyz2sph(points_xyz):
    ''' Transfer points coordinates from cartesian coordinate to spherical coordinate

    Parameters:
    points_xyz: array, shape == (n, 3)
    
    Return:
    points_sph: array, shape == (n, 3)
        the coord of each point is (r, theta, phi)
        where definition is same as Scipy:
        theta is the azimuthal angle (0~2pi) and phi is the polar angle (0~pi)
    '''
    epsilon=1e-100
    x, y, z = points_xyz[:,0], points_xyz[:,1], points_xyz[:,2]
    r = np.linalg.norm(points_xyz, axis=1)
    phi = np.arccos(z / (r+epsilon))
    theta = np.arctan2(y, x) # range [-pi, pi]
    theta = theta + (1-np.sign(np.sign(theta)+1))*2*np.pi # convert range to [0, 2pi]
    points_sph = np.vstack((r, theta, phi)).T
    return points_sph

# ！！！需要修改 ！！！
'''
def sph2xyz(points_sph):
    r, theta, phi = points_sph[:,0], points_sph[:,1], points_sph[:,2]
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    points_xyz = np.vstack((x, y, z)).T
    return points_xyz
'''

# ！！！需要修改 ！！！
# transfer points coordinates from cartesian coordinate to cylindrical coordinate
# points_xyz must be array([[x0,y0,z0], [x1,y1,z1], [x1,y1,z1], ...])
# returned points_sph is array([[r0, phi0, z0], [r1, phi1, z1], [r2, phi2, z2], ...])
'''
def xyz2cyl(points_xyz):
    r = np.linalg.norm(points_xyz[:,:2], axis=1)
    phi = np.arctan2(points_xyz[:,1], points_xyz[:,0])
    z = points_xyz[:,2]
    points_cyl = np.vstack((r, phi, z)).T
    return points_cyl
'''


if __name__ == "__main__":
    '''
    points_xyz = np.array([
        [1,0,-1]
    ])
    print(xyz2sph(points_xyz))
    '''
    # test
    points = 100*np.random.random((1000, 3))
    lmax = 30
    q = np.linspace(0.1, 1, num=100)
    f = np.ones(points.shape[0])
    print(intensity(q, points, f, lmax))