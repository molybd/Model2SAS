# to generate a 3D model from a mathematical description
# for example: a hollow sphere is "x**2+y**2+z**2 >= R1**2 and x**2+y**2+z**2 <= R2**2
# also, in spherical coordinates, a hollow sphere is r >=R1 and r <= R2
#
# function must return a python boolean type 
#   - True  |for point in the model
#   - False |for point not in the model
# boundaryList is 
#   - [xmin, xmax, ymin, ymax, zmin, zmax]
# coord is 
#   - 'xyz' |in (x,y,z)
#   - 'sph' |in (r, theta, phi)|theta: 0~pi ; phi: 0~2pi
#   - 'cyl' |in (r, phi, z)|phi:0-2pi
# coord must be the last arguement

def shell(point_sph, R1=8, R2=10, coord='sph'):
    r, theta, phi = point_sph[0], point_sph[1], point_sph[2]
    if r <=R2 and r >= R1:
        return True
    else:
        return False

def ring(point_sph, coord='sph'):
    r, theta, phi = point_sph[0], point_sph[1], point_sph[2]
    if r <=10 and r >= 8 and theta>=1.047 and theta <= 2.094:
        return True
    else:
        return False

def sphere(point_sph, R=50, coord='sph'):
    r, theta, phi = point_sph[0], point_sph[1], point_sph[2]
    if r <= R:
        return True
    else:
        return False

def cylinder(point_cyl, R=5, L=100, coord='cyl'):
    r, phi, z = point_cyl[0], point_cyl[1], point_cyl[2]
    if r <= R and abs(z) <= L/2:
        return True
    else:
        return False