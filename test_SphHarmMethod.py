import numpy as np
from matplotlib import pyplot
from scipy.special import sph_harm, spherical_jn

def xyz2sph(point):
    epsilon=1e-100
    r = np.linalg.norm(point)
    theta = np.arccos(point[2] / (r+epsilon))
    phi = np.arctan(point[1] / (point[0]+epsilon))
    return np.array([r, theta, phi]) # theta: 0~pi ; phi: 0~2pi

def func(q, point, lmax=5):
    p_sph = xyz2sph(point)
    r, theta, phi = p_sph[0], p_sph[1], p_sph[2]

    I = 0
    for l in range(lmax+1):
        for m in range(-l, l+1):
            I += abs(complex(0,1)**l)**2 * spherical_jn(l, q*r)**2 * sph_harm(m, l, phi, theta)**2

    return I

qlst = np.linspace(0.01, 1, num=100)
points = np.loadtxt('shell_pointcloud_numpy.txt')
Ilst = []
for q in qlst:
    Iq = 0
    for p in points:
        Iq += func(q, p, lmax=10)
    Ilst.append(Iq)

ax = pyplot.subplot(111)
ax.set_xscale("log", nonposx='clip')
ax.set_yscale("log", nonposy='clip')
pyplot.plot(qlst, Ilst)
pyplot.show()

