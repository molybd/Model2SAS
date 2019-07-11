import numpy as np
from matplotlib import pyplot
from scipy.special import sph_harm, spherical_jn, jv

def xyz2sph(point):
    epsilon=1e-100
    r = np.linalg.norm(point)
    theta = np.arccos(point[2] / (r+epsilon))
    phi = np.arctan(point[1] / (point[0]+epsilon))
    return np.array([r, theta, phi]) # theta: 0~pi ; phi: 0~2pi

def func(q, point, lmax=20):
    p_sph = xyz2sph(point)
    r, theta, phi = p_sph[0], p_sph[1], p_sph[2]

    I = 0
    for l in range(lmax+1):
        for m in range(-l, l+1):
            I += spherical_jn(l, q*r) * sph_harm(m, l, theta, phi)
            # I += (jv(1.5, q*R)/np.power((q*R),1.5)) * spherical_jn(l, q*r) * sph_harm(m, l, theta, phi)
    I = 4 * np.pi() * np.power(complex(0,1), l) * I
    return I



R = 1

qlst = np.linspace(0.01, 1, num=50)
points = np.loadtxt('shell_pointcloud_numpy.txt')
Ilst = []
for q in qlst:
    Iq = 0
    for p in points:
        Iq += func(q, p)**2
    Ilst.append(Iq)
'''
for q in qlst:
    Iq = 0
    for p1 in points:
        for p2 in points:
            Iq += func2(q, p1, p2, lmax=5)
    Ilst.append(Iq)
'''

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