'''some useful utility functions
'''

import numpy as np


def convert_coord(u:np.ndarray, v:np.ndarray, w:np.ndarray, source_coord:str, target_coord:str):
    ''' Convert coordinates
    car: Cartesian coordinates, in (x, y, z)
    sph: spherical coordinates, in (r, theta, phi) | theta: 0~2pi ; phi: 0~pi
    cyl: cylindrical coordinates, in (rho, phi, z) | theta:0-2pi

    Attributes:
        u: ndarray, 1st coord
        v: ndarray, 2nd coord
        w: ndarray, 3rd coord
        source_coord: 'car' or 'sph' or 'cyl'
        target_coord: 'car' or 'sph' or 'cyl'
    
    Return:
        converted u, v, w
    '''
    def car2sph(x:np.ndarray, y:np.ndarray, z:np.ndarray):
        '''convert cartesian coordinates to spherical coordinates
        '''
        epsilon=1e-100
        r = np.sqrt(x**2 + y**2 + z**2)
        phi = np.arccos(z / (r+epsilon))
        theta = np.arctan2(y, x) # range [-pi, pi]
        theta = theta + (1-np.sign(np.sign(theta)+1))*2*np.pi # convert range to [0, 2pi]
        return r, theta, phi
    def car2cyl(x:np.ndarray, y:np.ndarray, z:np.ndarray):
        '''convert cartesian coordinates to cylindrical coordinates
        '''
        rho = np.sqrt(x**2+y**2)
        theta = np.arctan2(y, x) # range [-pi, pi]
        theta = theta + (1-np.sign(np.sign(theta)+1))*2*np.pi # convert range to [0, 2pi]
        return rho, theta, z
    def sph2car(r:np.ndarray, theta:np.ndarray, phi:np.ndarray):
        '''convert spherical coordinates to cartesian coordinates
        '''
        x = r * np.cos(theta) * np.sin(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(phi)
        return x, y, z
    def cyl2car(rho:np.ndarray, theta:np.ndarray, z:np.ndarray):
        '''convert cylindrical coordinates to cartesian coordinates
        '''
        x = rho * np.cos(theta)
        y = rho * np.sin(theta)
        return x, y, z

    # all convert to Cartesian coordinates
    if source_coord == 'sph':
        x, y, z = sph2car(u, v, w)
    elif source_coord == 'cyl':
        x, y, z = cyl2car(u, v, w)
    else:
        x, y, z = u, v, w
    
    # then convert to desired coordinates
    if target_coord == 'car':
        return x, y, z
    elif target_coord == 'sph':
        return car2sph(x, y, z)
    elif target_coord == 'cyl':
        return car2cyl(x, y, z)


def gen_suggested_parameters(boundary_min:np.ndarray, boundary_max:np.ndarray) -> tuple:
    '''To evaluate optimal parameter values based on my experience.
    e.g. smaller interval surely gives better results but maybe unnecessary
    and may cause error since it uses much larger RAM and computing time. 
    
    Parameters:
        boundary_min
        boundary_max
    Returns:
        i: interval
        qmin: min q value
        qmax: max q value
    '''
    # i means interval here
    length = np.max(boundary_max-boundary_min)
    qmin, qmax = (1/length)*0.2, (1/length)*25
    smin, smax = qmin/(2*np.pi), qmax/(2*np.pi)
    i_expected_max = 1/(2*smax)
    i_times_ns_expected_min = 1/smin
    ns = 600  # grid number after fft, same as n_s in fft method
    i_expected_min = i_times_ns_expected_min / ns
    i = min(i_expected_min, i_expected_max)
    return i, qmin, qmax  # i means interval here