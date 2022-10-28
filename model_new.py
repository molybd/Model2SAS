'''All based on pytorch instead of numpy.
For the convenience of using CUDA
'''

import os
import sys

from stl import mesh
import numpy as np
import torch
from torch import Tensor

import calc_func_new as calc_func
from utility_new import timer, convert_coord


# def print_time(timestamp):
#     t = time.time()
#     print(t-timestamp)
#     return t


class Model:
    '''Parent model of part or assembly.
    TODO
    - 写gen_reciprocal_lattice方法
    - 写calc_sas1d方法
    '''

    def get_bound(self) -> tuple[tuple, tuple]:
        '''Get boundary of part model. Return 2 points which determine a cuboid fully containing the whole part model.
        To be overwritten.
        '''
        bound_min, bound_max = (0,0,0), (0,0,0)
        self.bound_min, self.bound_max = bound_min, bound_max
        return bound_min, bound_max

    def gen_lattice_meshgrid(self, spacing: float) -> tuple[Tensor, Tensor, Tensor]:
        '''Generate equally spaced meshgrid in 3d real space.
        '''
        bound_min, bound_max = self.get_bound()

        # ensure equally spacing lattice
        xmin, ymin, zmin = bound_min
        xmax, ymax, zmax = bound_max
        xnum = int((xmax-xmin)/spacing)+2
        xmax = xmin + spacing*(xnum-1)
        ynum = int((ymax-ymin)/spacing)+2
        ymax = ymin + spacing*(ynum-1)
        znum = int((zmax-zmin)/spacing)+2
        zmax = zmin + spacing*(znum-1)

        x1d = torch.linspace(xmin, xmax, xnum)
        y1d = torch.linspace(ymin, ymax, ynum)
        z1d = torch.linspace(zmin, zmax, znum)
        x, y, z = torch.meshgrid(x1d, y1d, z1d, indexing='ij')
        self.real_spacing = spacing
        self.x, self.y, self.z = x, y, z
        return x, y, z


    def gen_sld_lattice(self, spacing: float) -> Tensor:
        '''Generate SLD lattice of this part model. Values at lattice points are corresponding SLD.
        To be overwritten.
        '''
        self.gen_lattice_meshgrid(spacing)
        sld_lattice = torch.Tensor()
        self.sld_lattice = sld_lattice
        return sld_lattice

    @timer
    def gen_reciprocal_lattice(self, n_s: int | None = None) -> Tensor:
        '''Generate reciprocal lattice using FFT method.
        '''
        # determine n_s in reciprocal space
        if n_s is None:
            bound_min, bound_max = self.get_bound()
            xmin, ymin, zmin = bound_min
            xmax, ymax, zmax = bound_max
            L = max(xmax-xmin, ymax-ymin, zmax-zmin)
            # use s in fft. s = q/(2*pi)
            smin = (1/L) / (2*torch.pi)
            d = self.real_spacing
            n_s = int(1/(smin*d))+1
            n_real_lattice = max(self.sld_lattice.size())
        n_s = max(n_s, n_real_lattice)

        # using rfft to save time. so only upper half (qz>=0)
        F_half = torch.fft.rfftn(self.sld_lattice, s=(n_s, n_s, n_s))
        F_half = torch.fft.fftshift(F_half, dim=(0,1))

        self.n_s = n_s
        s1d = torch.fft.fftfreq(n_s, d=self.real_spacing)
        self.s1d = torch.fft.fftshift(s1d)
        self.s1dz = torch.fft.rfftfreq(n_s, d=self.real_spacing)
        self.F_half = F_half
        return F_half

    @timer
    def calc_sas1d(self, q1d: Tensor, orientation_average_offset: int = 100) -> tuple[Tensor, Tensor]:
        '''Calculate 1d SAS curve from reciprocal lattice
        '''
        s_input = q1d/(2*np.pi)
        s = s_input[torch.where((s_input>=self.s1d.min())&(s_input<=self.s1d.max()))]
        I_half = torch.real(self.F_half)**2 + torch.imag(self.F_half)**2

        # generate coordinates to interpolate using fibonacci grid
        # 每一个q值对应的球面取多少个取向进行平均
        n_on_sphere = s#**2 # increase too fast if use quadratic... time-consuming and unnecessary
        n_on_sphere = torch.round(n_on_sphere/n_on_sphere[0]) + orientation_average_offset
        sampling_x, sampling_y, sampling_z = calc_func.sampling_points(s, n_on_sphere)

        #### interpolate
        # 因为用的rfft，只有z>=0那一半，因此要将z<0的坐标转换为中心对称的坐标
        sign = torch.ones(sampling_z.size())
        sign[sampling_z<0] = -1.
        sampling_x, sampling_y, sampling_z = sign*sampling_x, sign*sampling_y, sign*sampling_z
        ds = self.s1d[1] - self.s1d[0]
        I_interp = calc_func.trilinear_interp(self.s1d, self.s1d, self.s1dz, I_half, ds, sampling_x, sampling_y, sampling_z)

        # orientation average
        I = []
        begin_index = 0
        for N in n_on_sphere:
            N = int(N)
            Ii = torch.sum(I_interp[begin_index:begin_index+N])/N
            I.append(Ii)
            begin_index += N
        I1d = torch.tensor(I)

        return 2*torch.pi*s, I1d


class Part(Model):
    '''Parent class for part model.
    Subclass: StlPart and MathPart
    '''
    def __init__(self, filename: str | None = None) -> None:
        '''Init function
        '''
        if filename is not None:
            self.filename = filename
            self.basename = os.path.basename(filename)
            self.partname = os.path.splitext(self.basename)[0]

    def import_sld_lattice(self, sld_lattice: Tensor, x: Tensor, y: Tensor, z: Tensor) -> None:
        self.real_spacing = x[1,0,0] - x[0,0,0]
        self.x, self.y, self.z = x, y, z
        self.sld_lattice = sld_lattice

    def get_bound(self) -> tuple[tuple, tuple]:
        xmin, ymin, zmin = self.x.min().item(), self.y.min().item(), self.z.min().item()
        xmax, ymax, zmax = self.x.max().item(), self.y.max().item(), self.z.max().item()
        bound_min, bound_max = (xmin, ymin, zmin), (xmax, ymax, zmax)
        self.bound_min, self.bound_max = bound_min, bound_max
        return bound_min, bound_max


class StlPart(Part):
    '''class for part from stl file.
    TODO
    - 重写get_boundary方法
    - 重写gen_sld_lattice方法
    '''
    def __init__(self, filename: str | None = None, sld_value: float = 1.0, centering : bool = True) -> None:
        '''load mesh from stl file
        '''
        super().__init__(filename=filename)
        if filename is not None:
            self.mesh = mesh.Mesh.from_file(self.filename)
            self.bound_min, self.bound_max = self.get_bound()
            if centering:# move model center to (0,0,0)
                center = self.mesh.get_mass_properties()[1]
                self.mesh.translate(-center)
        self.sld_value = sld_value

    def get_bound(self) -> tuple[tuple, tuple]:
        '''TODO
        - 是否需要扩展一些范围
        '''
        vec = self.mesh.vectors
        vec = vec.reshape((vec.shape[0]*vec.shape[1], vec.shape[2]))
        bound_min = vec.min(axis=0)
        bound_max = vec.max(axis=0)
        bound_min, bound_max = tuple(bound_min.tolist()), tuple(bound_max.tolist())
        self.bound_min, self.bound_max = bound_min, bound_max
        return bound_min, bound_max

    @timer
    def gen_sld_lattice(self, spacing: float) -> Tensor:
        '''Generate sld_lattice, which is sld value on each
        lattice meshgrid point. From stl mesh.
        '''
        x, y, z = self.gen_lattice_meshgrid(spacing)
        origins = torch.stack(
            [x.flatten(), y.flatten(), z.flatten()], dim=1
        )
        ray = torch.rand(3) - 0.5
        vectors = self.mesh.vectors.copy()
        triangles = torch.from_numpy(vectors).to(torch.float32)
        intersect_count = calc_func.moller_trumbore_intersect_count(origins, ray, triangles)
        index = intersect_count % 2   # 1 is in, 0 is out
        sld_lattice = self.sld_value * index
        sld_lattice = sld_lattice.reshape(x.size())

        self.sld_lattice = sld_lattice
        return sld_lattice


class MathPart(Part):
    '''class for part from math description.
    TODO
    - 重写get_boundary方法
    - 重写gen_sld_lattice方法
    '''
    def __init__(self, filename: str | None = None) -> None:
        '''load math object from py file
        TODO: or directly pass through?
        '''
        super().__init__(filename)
        if filename is not None:
            abspath = os.path.abspath(filename)
            dirname = os.path.dirname(abspath)
            sys.path.append(dirname)
            module = __import__(self.partname)
            self.math_description = module.MathDescription()

    def get_bound(self) -> tuple[tuple, tuple]:
        '''get bound from math_description
        '''
        bound_min, bound_max = self.math_description.get_bound()
        bound_min, bound_max = tuple(bound_min), tuple(bound_max)
        self.bound_min, self.bound_max = bound_min, bound_max
        return bound_min, bound_max

    @timer
    def gen_sld_lattice(self, spacing: float) -> Tensor:
        '''Generate sld_lattice, which is sld value on each
        lattice meshgrid point. From math description.
        '''
        x, y, z = self.gen_lattice_meshgrid(spacing)
        part_coord = self.math_description.coord
        u, v, w = convert_coord(x, y, z, 'car', part_coord)
        sld_lattice = self.math_description.sld(u, v, w)
        if isinstance(sld_lattice, np.ndarray):
            sld_lattice = torch.from_numpy(sld_lattice).to(torch.float32)

        self.sld_lattice = sld_lattice
        return sld_lattice



class Assembly(Model):
    '''Assembly of several part model
    TODO
    - consider transform of each model
    '''
    def __init__(self) -> None:
        '''
        '''
        self.transform = {}

    def translate(self, vector: tuple) -> None:
        '''translate model by vector.
        Parameters:
            vector: (x, y, z)
        '''
        self.transform['translate'] = dict(vector=vector)

    def rotate(self, axis: tuple, angle: float) -> None:
        '''rotate model around an axis passing origin by angle.
        Parameters:
            axis: vector describing direction of the axis
            angle: float, rotation angle in radians
        '''
        self.transform['rotate'] = dict(axis=axis, angle=angle)

    # 以后再说
    # def scaling(self, factor:float=None) -> None:
    #     '''scaling model around origin by factor.
    #     '''
    #     self.transform['scaling'] = dict(factor=factor)


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    @timer
    def main():
        # part = StlPart(filename=r'models\torus.stl')
        part = MathPart(filename=r'mathpart_template.py')
        part.gen_sld_lattice(spacing=2)
        part.gen_reciprocal_lattice()
        # print(part.F_half.size())
        q = torch.linspace(0.001, 1, 200)
        q, I = part.calc_sas1d(q)

        figure = plt.figure()
        ax = figure.add_subplot(projection='3d')
        x = part.x[np.where(part.sld_lattice!=0)]
        y = part.y[np.where(part.sld_lattice!=0)]
        z = part.z[np.where(part.sld_lattice!=0)]
        ax.scatter(x, y, z, c=part.sld_lattice[np.where(part.sld_lattice!=0)])
        plt.show()

        plt.plot(q, I)
        plt.xscale('log')
        plt.yscale('log')
        plt.show()
        
    main()


