# to generate a 3D model from a mathematical description
# for example: a hollow sphere is "x**2+y**2+z**2 >= R1**2 and x**2+y**2+z**2 <= R2**2
# also, in spherical coordinates, a hollow sphere is r >=R1 and r <= R2
# function must return a python boolean type !
# boundaryList is [xmin, xmax, ymin, ymax, zmin, zmax]
# coord is 'xyz'(x,y,z) or 'sph'(r, theta, phi)|theta: 0~pi ; phi: 0~2pi

def shell(point_sph, R1=8, R2=10):
    r, theta, phi = point_sph[0], point_sph[1], point_sph[2]
    if r <=R2 and r >= R1:
        return True
    else:
        return False

def ring(point_sph):
    r, theta, phi = point_sph[0], point_sph[1], point_sph[2]
    if r <=10 and r >= 8 and theta>=1.047 and theta <= 2.094:
        return True
    else:
        return False