# -*- coding: UTF-8 -*-

import numpy as np
from scipy.special import sph_harm, spherical_jn, jv
from multiprocessing import cpu_count
from p_tqdm import p_map


def printTime(last_timestamp, item):
    now = time.time()
    print('{:>10} {:^10}'.format(item, round(now-last_timestamp, 4)))
    timestamp = time.time()
    return timestamp


def intensity(q, points, f, lmax):

    def jl(q, r, l):
        r_ext1 = np.stack([r]*n_l, axis=-1)  # (r,) -> (r, l)
        r_ext1 = np.stack([r_ext1]*n_q, axis=-1)  # (r, l) -> (r, l, q)
        q_ext1 = np.stack([q]*n_l, axis=0)  # (q,) -> (l, q)
        q_ext1 = np.stack([q_ext1]*n_r, axis=0)  # (l, q) -> (r, l, q)
        rq_ext1 = r_ext1 * q_ext1   # (r, l, q)
        l_ext1 = np.stack([l]*n_r, axis=0)  # (l,) -> (r, l)
        l_ext1 = np.stack([l_ext1]*n_q, axis=-1)  # (r, l) -> (r, l, q)
        jl = spherical_jn(l_ext1, rq_ext1)  # (r, l, q)
        return jl.astype('float32')  # (r, l, q)

    def Ylm(l_ext, m, theta, phi):
        theta_ext2 = np.stack([theta]*n_m, axis=-1)  # (r,) -> (r, m)
        phi_ext2 = np.stack([phi]*n_m, axis=-1)  # (r,) -> (r, m)
        l_ext2 = np.stack([l_ext]*n_r, axis=0)  # (m,) -> (r, m)
        m_ext2 = np.stack([m]*n_r, axis=0)  # (m,) -> (r, m)
        Ylm = sph_harm(m_ext2, l_ext2, theta_ext2, phi_ext2)  # (r, m)
        return Ylm.astype('complex64')  # (r, m)

    def Sigma(f_ext3, jl_ext3, Ylm_ext1):
        Ylm_ext3 = np.stack([Ylm_ext1]*n_q, axis=-1)  #(r, m) -> (r, m, q)
        #timestamp1 = printTime(timestamp, 'Ylm_ext3')
        Sigma1 = f_ext3 * jl_ext3 * Ylm_ext3  # (r, m, q)
        #timestamp1 = printTime(timestamp1, 'f_ext3')
        return np.sum(Sigma1, axis=0)  # (m, q)


    q = q.astype('float32')
    q = q.reshape(q.size)  # (q,)
    points_sph = xyz2sph(points)
    r, theta, phi = points_sph[:,0], points_sph[:,1], points_sph[:,2]
    r, theta, phi = r.astype('float32'), theta.astype('float32'), phi.astype('float32')  # (r,)
    r, theta, phi = r.reshape(r.size), theta.reshape(theta.size), phi.reshape(phi.size)

    f = f.astype('float32')
    f = f.reshape(f.size)   # (r,)


    # _ext means extended
    # TIPS: use np.stack() to expand the dimension of array

    #timestamp0 = time.time()

    lmax = int(lmax)
    l = np.linspace(0, lmax, num=lmax+1, endpoint=True, dtype='int16')  # (l,)
    l_ext, m = [], []
    for li in l:
        for mi in range(-li, li+1):
            l_ext.append(li)
            m.append(mi)
    l_ext = np.array(l_ext, dtype='int16')  # (m,)
    m = np.array(m, dtype='int16')  # (m,)
    #timestamp = printTime(timestamp0, 'l&l_ext&m')

    n_r, n_l, n_m, n_q = r.size, l.size, m.size, q.size

    jl_ext1 = jl(q, r, l)  # (r, l, q)
    #timestamp = printTime(timestamp, 'jl_ext1')

    Ylm_ext1 = Ylm(l_ext, m, theta, phi)  # (r, m)
    #timestamp = printTime(timestamp, 'Ylm_ext1')

    # 接下来把各个部分都扩展成 shape=(r, m, q)
    # 尽量避免使用python循环嵌套，太慢了！！

    # 这一步使用一个循环比使用np.dot快得多
    f_ext3 = np.stack([f]*n_m, axis=-1)  # (r,) -> (r, m)
    f_ext3 = np.stack([f_ext3]*n_q, axis=-1)  # (r, m) -> (r, m, q)
    #timestamp = printTime(timestamp, 'f_ext3')

    # (r, l, q) -> (r, m, q)
    jl_ext3 = []
    for i in range(n_r):
        temp = []
        for j in range(n_l):
            temp += [ jl_ext1[i,j,:] ]*(2*j+1)  # (m, q)
        jl_ext3.append(temp)
    jl_ext3 = np.array(jl_ext3, dtype='float32')  # (r, m, q)
    #timestamp = printTime(timestamp, 'jl_ext3')

    Sigma1 = Sigma(f_ext3, jl_ext3, Ylm_ext1)
    #timestamp = printTime(timestamp, 'Sigma1')

    il = complex(0,1)**l  # (l,)
    il_ext4 = []
    for i in range(n_l):
        il_ext4 += [il[i]]*(2*i+1)
    # il_ext4.shape == (m,)
    il_ext5 = np.array([il_ext4_i*np.ones_like(q) for il_ext4_i in il_ext4], dtype='complex64')  # (m, q)
    #timestamp = printTime(timestamp, 'il_ext5')

    Alm = il_ext5 * Sigma1  # (m, q)
    #timestamp = printTime(timestamp, 'Alm3')
    I = 16 * np.pi**2 * np.sum(np.absolute(Alm)**2, axis=0)  # (q,)
    #timestamp = printTime(timestamp, 'I')

    return I


def intensity_parallel(q, points, f, lmax, cpu_usage=0.6, proc_num=None):
    # 本来是每一个q一个进程，但是在这里我希望把q切的不那么细，这样的话就不至于在建立进程上开销太大
    # 目前想的策略是切成并行进程数的4倍左右，但是每一个切片内q的数目在10~20比较好吧大概
    # 太大了会占用太多内存，太小了又会在建立进程上开销太大
    # 具体的值还得再试试

    # 确定proc_num
    if proc_num:
        proc_num = int(proc_num)
    else:
        proc_num = round(cpu_usage*cpu_count())
    # 确定切片的数目
    slice_length = 10
    k = round(q.size/slice_length/proc_num)
    slice_num = k * proc_num
    slice_length = np.ceil(q.size/slice_num)
    
    slice_num, slice_length = int(slice_num), int(slice_length)
    slice_index_begin = [i*slice_length for i in range(slice_num)]
    slice_index_end = [(i+1)*slice_length for i in range(slice_num)]
    q_list = []
    for i in range(slice_num):
        q_list.append(q[slice_index_begin[i]:slice_index_end[i]])
    #这里使用p_tqdm库来实现多进程下的进度条
    I_list= p_map(intensity, q_list, [points]*slice_num, [f]*slice_num, [lmax]*slice_num, num_cpus=proc_num)
    I = np.array(I_list, dtype='float32')
    I = I.reshape(I.size)
    return I



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


def coordConvert(points, source_coord, target_coord):
    ''' Convert coordinates

    Attributes:
        points: ndarray, shape==(n, 3)
        source_coord: 'xyz' or 'sph' or 'cyl'
        target_coord: 'xyz' or 'sph' or 'cyl'
    
    Return: 
        same shape as points
    '''
    def xyz2sph(points_xyz):
        epsilon=1e-100
        x, y, z = points_xyz[:,0], points_xyz[:,1], points_xyz[:,2]
        r = np.linalg.norm(points_xyz, axis=1)
        phi = np.arccos(z / (r+epsilon))
        theta = np.arctan2(y, x) # range [-pi, pi]
        theta = theta + (1-np.sign(np.sign(theta)+1))*2*np.pi # convert range to [0, 2pi]
        points_sph = np.vstack((r, theta, phi)).T
        return points_sph
    def xyz2cyl(points_xyz):
        x, y, z = points_xyz[:,0], points_xyz[:,1], points_xyz[:,2]
        rho = np.sqrt(x**2+y**2)
        theta = np.arctan2(y, x) # range [-pi, pi]
        theta = theta + (1-np.sign(np.sign(theta)+1))*2*np.pi # convert range to [0, 2pi]
        points_cyl = np.vstack((rho, theta, z)).T
        return points_cyl
    def sph2xyz(points_sph):
        r, theta, phi = points_sph[:,0], points_sph[:,1], points_sph[:,2]
        x = r * np.cos(theta) * np.sin(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(phi)
        points_xyz = np.vstack((x, y, z)).T
        return points_xyz
    def cyl2xyz(points_cyl):
        rho, theta, z = points_cyl[:,0], points_cyl[:,1], points_cyl[:,2]
        x = rho * np.cos(theta)
        y = rho * np.sin(theta)
        points_xyz = np.vstack((x, y, z)).T
        return points_xyz

    # 先全部转换成直角坐标系
    if source_coord != 'xyz':
        if source_coord == 'sph':
            points_xyz = sph2xyz(points)
        elif source_coord == 'cyl':
            points_xyz = cyl2xyz(points)
    else:
        points_xyz = points
    
    # 再转换为目标坐标系
    if target_coord == 'xyz':
        return points_xyz
    elif target_coord == 'sph':
        return xyz2sph(points_xyz)
    elif target_coord == 'cyl':
        return xyz2cyl(points_xyz)
    

if __name__ == "__main__":
    import time
    
    points_with_sld = np.loadtxt('test_points_with_sld.txt')
    points, f = points_with_sld[:,:3], points_with_sld[:,3]
    f = f.reshape(f.size)
    
    lmax = 50
    q = np.linspace(0.01, 1, num=200)

    begintime = time.time()
    #print('{:^10}|{:^10}'.format('item', 'time/sec'))
    #I = intensity(q, points, f, lmax)
    I = intensity_parallel(q, points, f, lmax, proc_num=1)
    endtime = time.time()
    print('total time: {} sec'.format(round(endtime-begintime, 2)))
    #print(I)
    from Plot import plotSasCurve
    plotSasCurve(q, I)

    