import time
import tqdm
import numpy as np
from scipy.special import sph_harm, spherical_jn

from utility import convert_coord


def printTime(last_timestamp:float, item:str) -> float:
    '''print time used from last timestamp
    '''
    now = time.time()
    print('{:>20} {:^10}'.format(item, round(now-last_timestamp, 4)))
    timestamp = time.time()
    return timestamp

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

    Alm0_without_jl = np.einsum('m,r,rm->rm', il, sld, Ylm)
    #timestamp = printTime(timestamp, 'Alm0')
    del il, Ylm
    ############################

    rq = np.einsum('r,q->rq', r, q)  #(r, q)
    rq = np.reshape(rq, (n_r, 1, n_q))  # (r, 1, q)
    l_ext1 = np.reshape(l, (1, n_l, 1))  # (1, l, 1)
    jl_ext1 = spherical_jn(l_ext1, rq)  # broadcast to (r, l, q)
    timestamp = printTime(timestamp, 'jl')
    # (r, l, q) -> (r, m, q)
    jl_ext1 = jl_ext1.astype(np.float32)

    jl = []
    for j in range(n_l):
        jl += [jl_ext1[:,j,:]]*(2*j+1)
    jl = np.array(jl, dtype=np.float32)
    jl = np.swapaxes(jl, 0, 1)  # (r, m, q)
    del rq, jl_ext1
    timestamp = printTime(timestamp, 'jl_rlq->rmq')

    Alm0_without_jl_real, Alm0_without_jl_imag = np.real(Alm0_without_jl), np.imag(Alm0_without_jl)
    Alm_real = np.einsum('rm,rmq->mq', Alm0_without_jl_real, jl)
    Alm_imag = np.einsum('rm,rmq->mq', Alm0_without_jl_imag, jl)
    #Alm = Alm_real + 1j*Alm_imag
    timestamp = printTime(timestamp, 'Alm')

    I = 16 * np.pi**2 * np.sum(Alm_real**2+Alm_imag**2, axis=0) #(q,)
    del jl, Alm_real, Alm_imag, Alm0_without_jl_real, Alm0_without_jl_imag
    return q, I