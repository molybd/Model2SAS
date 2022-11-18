'''All based on pytorch instead of numpy.
For the convenience of using CUDA
'''

import os
import sys

from stl import mesh
import numpy as np
import torch
from torch import Tensor

import calc_func
from utility import timer, convert_coord, abi2modarg, modarg2abi


# def print_time(timestamp):
#     t = time.time()
#     print(t-timestamp)
#     return t

class Model:
    '''Parent class for model, including Part and Assembly,
    which should overwrite get_F_value() and get_smax()
    method. Then, both can be treated equally in SAS calculation.
    '''
    def __init__(self, device: str = 'cpu') -> None:
        self.device = device

    def get_F_value(self, reciprocal_coord: Tensor) -> Tensor:
        '''Core method for model class, including Part and
        Assembly. Called by Sas class to calculate SAS pattern
        or curve.
        Parameters:
            reciprocal_coord: shape=(n, 3)
        To be overwritten.
        '''
        F_value = torch.ones(reciprocal_coord.shape[0], dtype=torch.complex64)
        return F_value

    def get_s_max(self) -> float:
        '''Return maximum s value of a part or assembly.
        '''
        smax = 0.
        return smax


class Part(Model):
    '''Parent class for part model.
    Subclass: StlPart and MathPart
    '''
    def __init__(self, filename: str | None = None, partname: str | None = None, device: str = 'cpu') -> None:
        '''Init function
        '''
        self.device = device
        self.partname = partname
        if filename is not None:
            self.filename = filename
            self.basename = os.path.basename(filename)
            if partname is None:
                self.partname = os.path.splitext(self.basename)[0]
        self.transformation = {}
        self.bound_min: tuple[float, float, float]
        self.bound_max: tuple[float, float, float]
        self.real_spacing: float
        self.x: Tensor
        self.y: Tensor
        self.z: Tensor
        self.x_original: Tensor
        self.y_original: Tensor
        self.z_original: Tensor
        self.sld: Tensor
        self.sld_original: Tensor
        self.centered: bool
        self.n_s: int
        self.s1d: Tensor
        self.s1dz: Tensor
        self.F_half: Tensor
        self.sx: Tensor
        self.sy: Tensor
        self.sz_half: Tensor

    def _get_suggested_config(self, Lmin: float, Lmax:float) -> tuple[float, int]:
        '''Calculate optimal real_spacing and n_s values
        for the generation of reciprocal spacing. Based on my
        experience and test, the meshgrid density is set to be
        40 on Lmax. But must ensure that the lattice number in
        Lmin is lerger than 10. n_s the larger the better, but
        may use too much RAM or VRAM, speed is also slow.

        Parameters:
            Lmin: minimum real space scale of a model
            Lmax: maximum real space scale of a model
        '''
        n_s = 501
        real_spacing = Lmax / 50
        real_spacing = min(real_spacing, Lmin/10)
        return real_spacing, n_s


    def set_real_lattice_meshgrid(self, x: Tensor, y: Tensor, z: Tensor) -> None:
        '''Import lattice meshgrid from outside of class.
        For the need of some other function.
        Must be evenly spaced in all 3 dimensions.
        '''
        spacing = x[1,0,0].item() - x[0,0,0].item()
        self.real_spacing = spacing
        self.x, self.y, self.z = x.to(self.device), y.to(self.device), z.to(self.device)

    def gen_real_lattice_meshgrid(self, spacing: float | None = None) -> tuple[Tensor, Tensor, Tensor]:
        '''Generate equally spaced meshgrid in 3d real space.
        '''
        bound_min, bound_max = self.get_bound()
        xmin, ymin, zmin = bound_min
        xmax, ymax, zmax = bound_max
        if spacing is None:
            Lmin = min(xmax-xmin, ymax-ymin, zmax-zmin)
            Lmax = max(xmax-xmin, ymax-ymin, zmax-zmin)
            spacing = self._get_suggested_config(Lmin, Lmax)[0]
        # print('real spacing: {}'.format(spacing))
        # ensure equally spacing lattice
        xmin, ymin, zmin = xmin+spacing/2, ymin+spacing/2, zmin+spacing/2
        xmax, ymax, zmax = xmax-spacing/2, ymax-spacing/2, zmax-spacing/2
        xnum = round((xmax-xmin)/spacing)+1
        xmax = xmin + spacing*(xnum-1)
        ynum = round((ymax-ymin)/spacing)+1
        ymax = ymin + spacing*(ynum-1)
        znum = round((zmax-zmin)/spacing)+1
        zmax = zmin + spacing*(znum-1)

        x1d = torch.linspace(xmin, xmax, xnum, device=self.device)
        y1d = torch.linspace(ymin, ymax, ynum, device=self.device)
        z1d = torch.linspace(zmin, zmax, znum, device=self.device)
        x, y, z = torch.meshgrid(x1d, y1d, z1d, indexing='ij')
        self.real_spacing = spacing
        self.x, self.y, self.z = x, y, z
        return x, y, z

    def set_real_lattice_sld(self, x: Tensor, y: Tensor, z: Tensor, sld: Tensor) -> None:
        '''To directly import a real lattice, jump over StlPart or MathPart,
        Mainly for test use.
        '''
        self.real_spacing = x[1,0,0].item() - x[0,0,0].item()
        self.x, self.y, self.z = x.to(self.device), y.to(self.device), z.to(self.device)
        self.sld = sld.to(self.device)
        self._store_original_real_lattice(x, y, z, sld)

    def get_bound(self) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        '''Get boundary of part model. Return 2 points which
        determine a cuboid fully containing the whole part model.
        To be overwritten.
        '''
        xmin, ymin, zmin = self.x.min().item(), self.y.min().item(), self.z.min().item()
        xmax, ymax, zmax = self.x.max().item(), self.y.max().item(), self.z.max().item()
        bound_min, bound_max = (xmin, ymin, zmin), (xmax, ymax, zmax)
        self.bound_min, self.bound_max = bound_min, bound_max
        return bound_min, bound_max

    def gen_real_lattice_sld(self) -> Tensor:
        '''Generate SLD lattice of this part model.
        Values at lattice points are corresponding SLD.
        Store original real lattice at last.
        To be overwritten.
        '''
        x, y, z, sld = self.x, self.y, self.z, self.sld
        self._store_original_real_lattice(x, y, z, sld)
        return sld

    def _store_original_real_lattice(self, x: Tensor, y: Tensor, z: Tensor, sld: Tensor) -> None:
        self.x_original = x.clone()
        self.y_original = y.clone()
        self.z_original = z.clone()
        self.sld_original = sld.clone()


    @timer
    def gen_reciprocal_lattice(self, n_s: int | None = None, need_centering: bool = True) -> Tensor:
        '''Generate reciprocal lattice from real lattice.
        The actual real lattice begins with bound_min, but
        FFT algorithm treats real space lattice as it begins
        with (0,0,0), which will make it different on
        reciprocal lattice but modulus will keeps the same.
        Attribute need_centering will change the reciprocal
        lattice to the right value.
        If no assembly needed, set need_centering to False
        to make it faster (about half the time). Otherwise,
        always keep it True is a better choice which takes
        less than 1s.
        ATTENTION:
            n_s controls precision in low q region. 
            larger n_s get more precise in low q, but may use
            too many memory since F increases in ^3. When n_s
            not specified, the calculation is controlled by
            scale factor in smin.
        '''
        # determine n_s in reciprocal space
        bound_min, bound_max = self.get_bound()
        xmin, ymin, zmin = bound_min
        xmax, ymax, zmax = bound_max
        if n_s is None:
            Lmin = min(xmax-xmin, ymax-ymin, zmax-zmin)
            Lmax = max(xmax-xmin, ymax-ymin, zmax-zmin)
            n_s = self._get_suggested_config(Lmin, Lmax)[1]
        n_real_lattice = max(self.sld.size())
        n_s = max(n_s, n_real_lattice)
        # print('n_s: {}'.format(n_s))
        # larger n_s get more precise in low q,
        # but may use too many memory since F increases in ^3

        # size at z (3rd dim) is different with x & y (dim1&2)
        s1d = torch.fft.fftfreq(n_s, d=self.real_spacing, device=self.device)
        s1d = torch.fft.fftshift(s1d)
        s1dz = torch.fft.rfftfreq(n_s, d=self.real_spacing, device=self.device)

        # using rfft to save time. so only upper half (qz>=0)
        F_half = torch.fft.rfftn(self.sld, s=(n_s, n_s, n_s))
        F_half = torch.fft.fftshift(F_half, dim=(0,1))

        # shift center to (0, 0, 0)
        # time-consuming part
        if need_centering:
            xmin, ymin, zmin = xmin+self.real_spacing/2, ymin+self.real_spacing/2, zmin+self.real_spacing/2
            F_half = self._translate_on_reciprocal_lattice(F_half, s1d, s1d, s1dz, xmin, ymin, zmin)
            self.centered = True
        else:
            self.centered = False

        # eliminate the difference caused by different spacing in real space.
        F_half = self.real_spacing**3 * F_half

        self.n_s = n_s
        self.s1d = s1d
        self.s1dz = s1dz
        self.F_half = F_half
        return F_half

    @timer
    def _translate_on_reciprocal_lattice(self, F: Tensor, sx_1d: Tensor, sy_1d: Tensor, sz_1d: Tensor, vx: float, vy:float, vz:float) -> Tensor:
        '''Default model center is at (0, 0, 0), referring to
        self.bound_min, self.bound_max. But real lattice bound
        for fft is (0, 0, 0) & self.bound_max-self.bound_min.
        So shift_vector = self.bound_min,
        shift_multiplier =  e^{-i * 2pi * s dot (-shif_vector)}.
        Shift_multiplier only change complex argument, so
        corresponding method is used here.
        Will generate reciprocal meshgrid sx, sy, sz.
        '''
        sx, sy, sz_half = torch.meshgrid(sx_1d, sy_1d, sz_1d, indexing='ij')
        multiplier_arg = -1 * 2*torch.pi * (sx*vx + sy*vy + sz_half*vz)
        F_mod, F_arg = abi2modarg(F)
        F_new = modarg2abi(F_mod, F_arg+multiplier_arg)
        # will raise error when calculating exp on complex numbers on cuda
        self.sx, self.sy, self.sz_half = sx, sy, sz_half
        return F_new

    @timer
    def translate(self, vector: tuple[float, float, float]) -> None:
        '''Translate model by vector.
        Change real space lattice directly;
        But in reciprocal space, change also depends on input
        coordinates. So transform methods like translate and
        rotate will return a closure function which will act
        on coordinates and reciprocal lattice in the later call.
        Method refer to self._translate_on_reciprocal_lattice()
        '''
        vx, vy, vz = vector
        # real space
        self.x = self.x + vx
        self.y = self.y + vy
        self.z = self.z + vz

        # reciprocal lattice
        multiplier_arg = -1 * 2*torch.pi * (self.sx*vx + self.sy*vy + self.sz_half*vz) # only imaginary part

        def apply_on_reciprocal_lattice(coord: Tensor, F: Tensor) -> tuple[Tensor, Tensor]:
            F_mod, F_arg = abi2modarg(F)
            F_new = modarg2abi(F_mod, F_arg+multiplier_arg)
            return coord, F_new

        index = len(self.transformation)
        self.transformation[index] = dict(
            type = 'translate',
            param = (vector,),
            func = apply_on_reciprocal_lattice  # 闭包函数
        )

    @timer
    def rotate(self, axis: tuple[float, float, float], angle: float) -> None:
        '''Rotate model around an axis passing origin by angle.
        Change real space lattice directly;
        But in reciprocal space, change also depends on input
        coordinates. So transform methods like translate and
        rotate will return a closure function which will act
        on coordinates and reciprocal lattice in the later call.
        Uses the Euler-Rodrigues formula. see
        https://en.wikipedia.org/wiki/Euler%E2%80%93Rodrigues_formula
        Parameters:
            axis: vector describing direction of the axis
            angle: float, rotation angle in radians
        '''
        def euler_rodrigues_rotate(coord: Tensor, axis_local: tuple[float, float, float], angle_local: float) -> Tensor:
            ax = torch.tensor(axis_local, device=self.device)
            ax = ax / torch.sqrt(torch.sum(ax**2))
            ang = torch.tensor(angle_local, device=self.device)
            a = torch.cos(ang/2)
            b = ax[0]*torch.sin(ang/2)
            c = ax[1]*torch.sin(ang/2)
            d = ax[2]*torch.sin(ang/2)
            w = torch.tensor((b, c, d), device=self.device)

            x = coord
            wx = -torch.linalg.cross(x, w, dim=-1)
            x_rotated = x + 2*a*wx + 2*(-torch.linalg.cross(wx, w, dim=-1))
            return x_rotated

        # real space
        points = torch.stack([self.x, self.y, self.z], dim=-1)
        points = euler_rodrigues_rotate(points, axis, angle)
        self.x, self.y, self.z = torch.unbind(points, dim=-1)

        # reciprocal lattice
        def apply_on_reciprocal_lattice(coord: Tensor, F: Tensor) -> tuple[Tensor, Tensor]:
            new_coord = euler_rodrigues_rotate(coord, axis, -angle) # rotate coord reversely to fit reciprocal lattice
            return new_coord, F

        index = len(self.transformation)
        self.transformation[index] = dict(
            type = 'rotate',
            param = (axis, angle),
            func = apply_on_reciprocal_lattice  # 闭包函数
        )

    def clear_transformation(self) -> None:
        '''Clear all transformations. Set part model to default.
        '''
        self.x, self.y, self.z = self.x_original.clone(), self.y_original.clone(), self.z_original.clone()
        self.transformation = {}

    @timer
    def get_F_value(self, reciprocal_coord: Tensor) -> Tensor:
        '''Get F value (scattering amplitude) of certain coordinates
        in reciprocal space.
        Considering transformations applied to the part model, which
        will be realized in reciprocal space. For now, 2 kinds of
        transformation are realized here: translate and rotate. Translate
        doesn't change the reciprocal meshgrid, but change the F value at
        each grid point; rotation, on the contrary, change the reciprocal
        meshgrid coordinates (rotate them), but doesn't change the F values.
        So, this function will return the F values of each coordinates
        after applying all transformations.

        Parameters:
            reciprocal_coord: shape=(n, 3)
        '''
        new_coord = reciprocal_coord.to(self.device)
        new_F_half = self.F_half.clone()

        # apply transform
        if self.centered:
            for transform in self.transformation.values():
                new_coord, new_F_half = transform['func'](new_coord, new_F_half)

        # 因为用的rfft，只有z>=0那一半，因此要将z<0的坐标转换为中心对称的坐标
        sx, sy, sz = torch.unbind(new_coord, dim=-1)
        sign = torch.ones(sx.size(), device=self.device)
        sign[sz<0] = -1.
        sx, sy, sz = sign*sx, sign*sy, sign*sz
        ds = self.s1d[1] - self.s1d[0]

        ################################################
        # 这里分了三种情况分别进行测试，分别是直接复数插值，实部虚部分别插值，
        # 以及将复数转换为模与辐角表示后分别进行插值。
        # 结果表明，直接复数插值和实部虚部分别插值的结果相同;
        # 而模与辐角分别插值得到的一维散射曲线与计算一维曲线中对强度进行插值结果
        # 相同（因为模实际上就是强度的开方）。
        # 但是在组合模型中，模与辐角分别插值导致组合模型的一维散射曲线低q区震动
        # 比较严重。这是因为组合模型更依赖于低q区准确的实部和虚部，而零件模型完全
        # 只依赖于模，所以精度差一些。
        ################################################
        

        # 直接复数插值
        F_value = calc_func.trilinear_interp(
            sx, sy, sz, self.s1d, self.s1d, self.s1dz, new_F_half, ds
        )

        # 模与辐角分别插值
        # new_F_half_mod, new_F_half_arg = abi2modarg(new_F_half)
        # F_value_mod = calc_func.trilinear_interp(
        #     sx, sy, sz, self.s1d, self.s1d, self.s1dz, new_F_half_mod, ds
        # )
        # F_value_arg = calc_func.trilinear_interp(
        #     sx, sy, sz, self.s1d, self.s1d, self.s1dz, new_F_half_arg, ds
        # )
        # F_value = modarg2abi(F_value_mod, F_value_arg)

        return F_value

    def get_s_max(self) -> float:
        '''Return maximum s value available for a part model.
        '''
        smax = min(torch.abs(self.s1d.min()).item(), torch.abs(self.s1d.max()).item())
        return smax

    def get_real_lattice(self, output_device: str = 'cpu') -> tuple[Tensor, Tensor, Tensor, Tensor]:
        '''Return real lattice
        '''
        x = self.x.to(output_device)
        y = self.y.to(output_device)
        z = self.z.to(output_device)
        sld = self.sld.to(output_device)
        return x, y, z, sld

    @timer
    def _calc_sas1d(self, q1d: Tensor, orientation_average_offset: int = 100) -> tuple[Tensor, Tensor]:
        '''Calculate 1d SAS curve from reciprocal lattice.
        For test only now. Will be transfered to sas module.
        '''
        q1d = q1d.to(self.device)
        s_input = q1d/(2*torch.pi)
        s = s_input[torch.where(s_input<=self.s1d.max())]

        # generate coordinates to interpolate using fibonacci grid
        # 每一个q值对应的球面取多少个取向进行平均
        n_on_sphere = s#**2 # increase too fast if use quadratic... time-consuming and unnecessary
        n_on_sphere = torch.round(n_on_sphere/n_on_sphere[0]) + orientation_average_offset
        sx, sy, sz = calc_func.sampling_points(s, n_on_sphere)

        #### interpolate
        reciprocal_coord = torch.stack([sx, sy, sz], dim=-1)
        F = self.get_F_value(reciprocal_coord)
        I = F.real**2 + F.imag**2

        # orientation average
        n_on_sphere = n_on_sphere.to('cpu')
        I_list = []
        begin_index = 0
        for N in n_on_sphere:
            N = int(N)
            Ii = torch.sum(I[begin_index:begin_index+N])/N
            I_list.append(Ii.to('cpu').item())
            begin_index += N
        I1d = torch.tensor(I_list)

        return 2*torch.pi*s.to('cpu'), I1d


class StlPart(Part):
    '''class for part from stl file.
    Rewrite get_bound and gen_real_lattice_sld methods.
    '''
    def __init__(self, filename: str | None = None, partname: str | None = None, sld_value: float = 1.0, centering : bool = True, device: str = 'cpu') -> None:
        '''load mesh from stl file
        '''
        super().__init__(filename=filename, partname=partname, device=device)
        if filename is not None:
            self.mesh = mesh.Mesh.from_file(self.filename)
            self.bound_min, self.bound_max = self.get_bound()
            if centering:# move model center to (0,0,0)
                center = self.mesh.get_mass_properties()[1]
                self.mesh.translate(-center)
        self.sld_value = sld_value

    def get_bound(self) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        '''Get boundary of part model. Return 2 points which
        determine a cuboid fully containing the whole part model.
        '''
        vec = self.mesh.vectors
        vec = vec.reshape((vec.shape[0]*vec.shape[1], vec.shape[2]))
        bound_min = vec.min(axis=0)
        bound_max = vec.max(axis=0)
        bound_min, bound_max = tuple(bound_min.tolist()), tuple(bound_max.tolist())
        self.bound_min, self.bound_max = bound_min, bound_max
        return bound_min, bound_max

    @timer
    def gen_real_lattice_sld(self) -> Tensor:
        '''Generate sld in real lattice, which is sld value on each
        lattice meshgrid point. From stl mesh.
        '''
        x, y, z = self.x, self.y, self.z
        origins = torch.stack(
            [x.flatten(), y.flatten(), z.flatten()],
            dim=1
        )
        ray = torch.rand(3) - 0.5
        ray = ray.to(self.device)
        vectors = self.mesh.vectors.copy()
        triangles = torch.from_numpy(vectors).to(torch.float32).to(self.device)
        intersect_count = calc_func.moller_trumbore_intersect_count(origins, ray, triangles)
        index = intersect_count % 2   # 1 is in, 0 is out
        sld = self.sld_value * index
        sld = sld.reshape(x.size())

        self.sld = sld
        self._store_original_real_lattice(x, y, z, sld)
        return sld


class MathPart(Part):
    '''class for part from math description.
    Rewrite get_bound and gen_real_lattice_sld methods.
    '''
    def __init__(self, filename: str | None = None, partname: str | None = None, device: str = 'cpu') -> None:
        '''load math object from py file
        TODO: or directly pass through?
        '''
        super().__init__(filename=filename, partname=partname, device=device)
        if filename is not None:
            abspath = os.path.abspath(filename)
            dirname = os.path.dirname(abspath)
            sys.path.append(dirname)
            module = __import__(self.partname)
            self.math_description = module.MathDescription()

    def get_bound(self) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        '''Get boundary of part model. Return 2 points which
        determine a cuboid fully containing the whole part model.
        '''
        bound_min, bound_max = self.math_description.get_bound()
        bound_min, bound_max = tuple(bound_min), tuple(bound_max)
        self.bound_min, self.bound_max = bound_min, bound_max
        return bound_min, bound_max

    @timer
    def gen_real_lattice_sld(self) -> Tensor:
        '''Generate sld in real lattice, which is sld value on each
        lattice meshgrid point. From math description.
        '''
        x, y, z = self.x, self.y, self.z
        part_coord = self.math_description.coord
        u, v, w = convert_coord(x, y, z, 'car', part_coord)
        sld = self.math_description.sld(u, v, w) # will deal with device in function automatically
        if isinstance(sld, np.ndarray):
            sld = torch.from_numpy(sld).to(torch.float32).to(self.device)

        self.sld = sld
        self._store_original_real_lattice(x, y, z, sld)
        return sld


class Assembly(Model):
    '''Assembly of several part model
    '''
    def __init__(self, *parts: Part, device: str = 'cpu') -> None:
        '''All part model must in same device.
        '''
        self.parts = {}
        self.add_parts(*parts)
        self.device = device

    def add_parts(self, *parts: Part) -> None:
        '''Add part object or list of part object to self.parts dictionary'''
        for p in parts:
            i = len(self.parts)
            self.parts[i] = p

    @timer
    def get_F_value(self, reciprocal_coord: Tensor) -> Tensor:
        '''Get F value (scattering amplitude) of certain coordinates
        in reciprocal space. Sum over all parts.
        '''
        F_value = torch.zeros(reciprocal_coord.shape[0], dtype=torch.complex64, device=self.device)
        for part in self.parts.values():
            F_value += part.get_F_value(reciprocal_coord).to(self.device)
        return F_value

    def get_s_max(self) -> float:
        '''Return maximum s value of assembly model.
        '''
        smax_list = []
        for part in self.parts.values():
            smax_list.append(part.get_s_max())
        return min(smax_list)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from plot import plot_parts, plot_sas1d

    # torch.set_default_dtype(torch.float64)

    @timer
    def main():
        # part = StlPart(filename=r'models/torus.stl', device='cuda')
        # part = MathPart(filename=r'models/cylinder_y.py', device='cpu')
        part = MathPart(filename=r'mathpart_template.py', device='cpu')
        part.gen_real_lattice_meshgrid()
        part.gen_real_lattice_sld()
        part.gen_reciprocal_lattice()

        # part.rotate((1,0,0), torch.pi/2)
        part.translate((30,0,0))

        # plot_parts(part, show=True)
    
        q = torch.linspace(0.005, 2.5, 200)
        q, I = part._calc_sas1d(q)
        plot_sas1d(q, I, show=True, savename='test.png')
        
    main()


