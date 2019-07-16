import numpy as np
from matplotlib import pyplot
from scipy.special import sph_harm, spherical_jn, jv

def xyz2sph(point):
    epsilon=1e-100
    r = np.linalg.norm(point)
    theta = np.arccos(point[2] / (r+epsilon))
    phi = np.arctan(point[1] / (point[0]+epsilon))
    return np.array([r, theta, phi]) # theta: 0~pi ; phi: 0~2pi

def Alm(q, points, l, m):
    A = 0
    for p in points:
        p_sph = xyz2sph(p)
        r, theta, phi = p_sph[0], p_sph[1], p_sph[2]
        A += spherical_jn(l, q*r) * sph_harm(m, l, phi, theta)
        # A += (jv(1.5, q*R)/(q*R)**1.5) * spherical_jn(l, q*r) * sph_harm(m, l, theta, phi)
    return 4 * np.pi * complex(0,1)**l * A

def Iq(q, points, lmax=20):
    I = 0
    for l in range(lmax+1):
        for m in range(-l, l+1):
            I += abs(Alm(q, points, l, m))**2
    return I

if __name__ == "__main__":

    R = 0.5

    qlst = np.linspace(0.01, 1, num=50)
    points = np.loadtxt('shell_pointcloud_numpy.txt')

    #这里可以直接带入数组进行计算
    Ilst = Iq(qlst, points, lmax=5)
    '''
    for q in qlst:
        Iq = 0
        for p1 in points:
            for p2 in points:
                Iq += func2(q, p1, p2, lmax=5)
        Ilst.append(Iq)
    '''

    qIarray = np.vstack((qlst, Ilst)).T
    np.savetxt('qI_sphharm.dat', qIarray)

    ax = pyplot.subplot(111)
    ax.set_xscale("log", nonposx='clip')
    ax.set_yscale("log", nonposy='clip')
    pyplot.plot(qlst, Ilst)
    pyplot.show()

    '''
    # 以下是画球的散射函数
    qlst = np.linspace(0.01, 1, num=50)
    Ilst = (jv(1.5, qlst*R)/(qlst*R)**1.5)**2
    ax = pyplot.subplot(111)
    ax.set_xscale("log", nonposx='clip')
    ax.set_yscale("log", nonposy='clip')
    pyplot.plot(qlst, Ilst)
    pyplot.show()
    '''