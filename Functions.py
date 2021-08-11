# -*- coding: UTF-8 -*-

import numpy as np
from scipy.special import sph_harm, spherical_jn
from multiprocessing import cpu_count, Pool
#from tqdm import tqdm
import psutil
import time

##### 下一步计划 #####
# 根据机器的内存和显存自动切片，防止爆内存或显存
####################


def printTime(last_timestamp, item):
    now = time.time()
    print('{:>20} {:^10}'.format(item, round(now-last_timestamp, 4)))
    timestamp = time.time()
    return timestamp        

def intensity_cpu(q, points, f, lmax, slice_num=None):

    q = q.astype('float32')
    q = q.reshape(q.size)  # (q,)
    points_sph = xyz2sph(points)
    r, theta, phi = points_sph[:,0], points_sph[:,1], points_sph[:,2]
    r, theta, phi = r.astype('float32'), theta.astype('float32'), phi.astype('float32')  # (r,)
    r, theta, phi = r.reshape(r.size), theta.reshape(theta.size), phi.reshape(phi.size)

    f = f.astype('float32')
    f = f.reshape(f.size)   # (r,)

    # _ext means extended
    # TIPS: use einsum() method

    lmax = int(lmax)
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
    #timestamp = printTime(timestamp, 'Ylm preparation')
    Ylm = sph_harm(m_ext2, l_ext2, theta_ext2, phi_ext2).astype(np.complex64)  # broadcast to (r, m) float64
    del theta_ext2, phi_ext2, l_ext2, m_ext2
    timestamp = printTime(timestamp, 'Ylm')
    #########################

    ##### calculate Alm0 without jl #####
    il = complex(0,1)**l  # (l,)
    il_list = []
    for i in range(n_l):
        il_list += [il[i]]*(2*i+1)
    il = np.array(il_list, dtype='complex64')  # (m,)
    #timestamp = printTime(timestamp, 'il')

    Alm0_without_jl = np.einsum('m,r,rm->rm', il, f, Ylm)
    #timestamp = printTime(timestamp, 'Alm0')
    del il, Ylm
    ############################

    ##### 切片循环，防止爆内存 #####
    free_memory = psutil.virtual_memory().free
    if slice_num:
        slice_num = int(slice_num)
    else:
        max_size = q.shape[0] * points.shape[0] * (lmax+1)**2 * 4  # 使用float32，每个数字4byte
        slice_num = int(max_size/(0.9*free_memory)) + 1
    slice_length = q.size // slice_num + 1
    I_list = []
    for i in range(slice_num):
        index_begin = i * slice_length
        index_end = (i+1) * slice_length
        qi = q[index_begin:index_end]

        ##### calculate jl #####
        # 使用l计算，然后再扩充，直接计算会特别慢
        n_qi = qi.size
        rq = np.einsum('r,q->rq', r, qi)  #(r, q)
        rq = np.reshape(rq, (n_r, 1, n_qi))  # (r, 1, q)
        l_ext1 = np.reshape(l, (1, n_l, 1))  # (1, l, 1)
        jl_ext1 = spherical_jn(l_ext1, rq)  # broadcast to (r, l, q)
        timestamp = printTime(timestamp, 'jl')
        # (r, l, q) -> (r, m, q)
        jl_ext1 = jl_ext1.astype(np.float32)

        # 目前最快的方法
        jl = []
        for j in range(n_l):
            jl += [jl_ext1[:,j,:]]*(2*j+1)
        jl = np.array(jl, dtype=np.float32)
        jl = np.swapaxes(jl, 0, 1)  # (r, m, q)
        del rq, jl_ext1
        '''
        # 用下面的方法不管是用numpy还是torch都是最慢的！
        jl = jl_ext1[:,0,:].reshape((n_r,1,n_q))
        for j in range(1, n_l):
            t = jl_ext1[:,j,:].reshape((n_r,1,n_q))
            jl = np.concatenate([jl]+[t]*(2*j+1), axis=1)

        # 下面的方法比上面的方法快不少，但是不是最快的
        jl = []
        for k in range(n_r):
            temp = []
            for j in range(n_l):
                temp += [ jl_ext1[k,j,:] ]*(2*j+1)  # (m, q)
            jl.append(temp)
        jl = torch.from_numpy(np.array(jl))  # (r, m, q)
        '''
        timestamp = printTime(timestamp, 'jl_rlq->rmq')
        #########################
 
        ##### calculate Alm #####
        Alm = np.einsum('rm,rmq->mq', Alm0_without_jl, jl)  # (m, q)
        timestamp = printTime(timestamp, 'Alm')
        #########################

        Ii = 16 * np.pi**2 * np.sum(np.absolute(Alm)**2, axis=0)  # (q,)
        I_list.append(Ii)
        #timestamp = printTime(timestamp, 'I_i')
        del jl, Alm  # 即时垃圾回收，不然内存会不够
        timestamp = printTime(timestamp, 'gc')

    I = np.hstack(I_list)
    timestamp = printTime(timestamp, 'I')

    return I

def intensity_gpu(q, points, f, lmax, slice_num=None):
    
    import torch
    
    q = q.astype('float32')
    q = q.reshape(q.size)  # (q,)
    points_sph = xyz2sph(points)
    r, theta, phi = points_sph[:,0], points_sph[:,1], points_sph[:,2]
    r, theta, phi = r.astype('float32'), theta.astype('float32'), phi.astype('float32')  # (r,)
    r, theta, phi = r.reshape(r.size), theta.reshape(theta.size), phi.reshape(phi.size)

    f = f.astype('float32')
    f = f.reshape(f.size)   # (r,)

    # _ext means extended
    # TIPS: use einsum() method

    lmax = int(lmax)
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
    #timestamp = printTime(timestamp, 'Ylm preparation')
    Ylm = sph_harm(m_ext2, l_ext2, theta_ext2, phi_ext2).astype(np.complex64)  # broadcast to (r, m) float64
    del theta_ext2, phi_ext2, l_ext2, m_ext2
    timestamp = printTime(timestamp, 'Ylm')
    #########################

    ##### calculate Alm0 without jl #####
    il = complex(0,1)**l  # (l,)
    il_list = []
    for i in range(n_l):
        il_list += [il[i]]*(2*i+1)
    il = np.array(il_list, dtype='complex64')  # (m,)
    #timestamp = printTime(timestamp, 'il')

    Alm0_without_jl = np.einsum('m,r,rm->rm', il, f, Ylm)
    #timestamp = printTime(timestamp, 'Alm0')
    del il, Ylm
    ############################

    ##### 切片循环，防止爆内存 #####
    free_memory = psutil.virtual_memory().free
    if slice_num:
        slice_num = int(slice_num)
    else:
        max_size = q.shape[0] * points.shape[0] * (lmax+1)**2 * 4  # float32 一个数占用4byte
        slice_num = int(max_size/(0.9*free_memory)) + 1
    slice_length = q.size // slice_num + 1
    I_list = []
    for i in range(slice_num):
        index_begin = i * slice_length
        index_end = (i+1) * slice_length
        qi = q[index_begin:index_end]

        ##### calculate jl #####
        # 使用l计算，然后再扩充，直接计算会特别慢
        n_qi = qi.size
        rq = np.einsum('r,q->rq', r, qi)  #(r, q)
        rq = np.reshape(rq, (n_r, 1, n_qi))  # (r, 1, q)
        l_ext1 = np.reshape(l, (1, n_l, 1))  # (1, l, 1)
        jl_ext1 = spherical_jn(l_ext1, rq)  # broadcast to (r, l, q)
        timestamp = printTime(timestamp, 'jl')
        # (r, l, q) -> (r, m, q)
        jl_ext1 = jl_ext1.astype(np.float32)

        # 目前最快的方法
        jl = []
        for j in range(n_l):
            jl += [jl_ext1[:,j,:]]*(2*j+1)
        jl = np.array(jl, dtype=np.float32)
        jl = np.swapaxes(jl, 0, 1)  # (r, m, q)
        del rq, jl_ext1
        timestamp = printTime(timestamp, 'jl_rlq->rmq')
        #########################

        ##### calculate Alm #####
        # 将实数与复数分开计算，内存占用比都变为复数计算大为减少，计算速度也快很多
        Alm0_without_jl_real_tensor = torch.from_numpy(np.real(Alm0_without_jl))
        Alm0_without_jl_imag_tensor = torch.from_numpy(np.imag(Alm0_without_jl))
        jl_tensor = torch.from_numpy(jl)
        Alm_real = torch.einsum('rm,rmq->mq', Alm0_without_jl_real_tensor, jl_tensor)
        Alm_imag = torch.einsum('rm,rmq->mq', Alm0_without_jl_imag_tensor, jl_tensor)
        Alm = Alm_real + 1j*Alm_imag
        '''
        # 这是直接计算的方法，因为torch.einsum计算实数和复数会报错，所以得先把jl变成complex64
        # 这就导致内存占用急剧增加
        Alm0_without_jl_tensor = torch.from_numpy(Alm0_without_jl)
        jl_tensor = torch.from_numpy(jl).type(torch.complex64)
        Alm = torch.einsum('rm,rmq->mq', Alm0_without_jl_tensor, jl_tensor)  # (m, q)
        '''
        Alm = Alm.numpy()
        timestamp = printTime(timestamp, 'Alm')
        #########################

        Ii = 16 * np.pi**2 * np.sum(np.absolute(Alm)**2, axis=0)  # (q,)
        I_list.append(Ii)
        #timestamp = printTime(timestamp, 'I_i')
        del jl, jl_tensor, Alm  # 即时垃圾回收，不然下一个循环内存会不够
        timestamp = printTime(timestamp, 'gc')

    I = np.hstack(I_list)
    timestamp = printTime(timestamp, 'I')

    return I

def intensity_cpu_parallel(q, points, f, lmax, core_num=2, proc_num=4):
    '''
    目前来看主要的瓶颈是内存不够，而不是CPU，因此其实意义不大
    但是以防有些有巨大内存的需要更快的计算速度，因此还是先保留
    未来也许会把这个方法删除
    '''
    if core_num > cpu_count():
        core_num = cpu_count()
    else:
        core_num = int(core_num)

    slice_num = int(proc_num)

    slice_length = round(q.size/slice_num)
    q_list = []
    for i in range(slice_num-1):
        q_list.append(q[i*slice_length:(i+1)*slice_length])
    q_list.append(q[(slice_num-1)*slice_length:])
    # 以防最后几个是空的，得去掉不然会报错
    while q_list[-1].size == 0:
        q_list.pop()
    slice_num = len(q_list)

    pool = Pool(core_num)
    args = zip(q_list, [points]*slice_num, [f]*slice_num, [lmax]*slice_num)
    result = pool.starmap_async(intensity_cpu, args)
    pool.close()
    pool.join()
    I = np.array(result.get()).flatten()

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
    
    points_with_sld = np.loadtxt('test_points_with_sld2.txt')
    points, f = points_with_sld[:,:3], points_with_sld[:,3]
    f = f.reshape(f.size)
    
    lmax = 50
    q = np.linspace(0.01, 1, num=200)

    begintime = time.time()
    #print('{:^10}|{:^10}'.format('item', 'time/sec'))
    I = intensity_cpu(q, points, f, lmax)
    #I = intensity_gpu(q, points, f, lmax)
    #I = intensity_cpu_parallel(q, points, f, lmax, core_num=2, proc_num=4)
    endtime = time.time()
    print('total time: {:.2f} sec'.format(endtime-begintime))
    #print(I)
    from Plot import plotSasCurve
    plotSasCurve(q, I)
