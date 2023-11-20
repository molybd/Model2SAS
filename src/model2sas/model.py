"""All based on pytorch instead of numpy.
For the convenience of using CUDA
"""

import os
import sys
import copy
from typing import Literal

from stl import mesh
import numpy as np
import torch
from torch import Tensor
import Bio.PDB
import periodictable as pt

from . import calcfunc
from .calcfunc import euler_rodrigues_rotate, convert_coord, abi2modarg, modarg2abi
from .utils import log, MathModelClassBase


class Model:
    """Parent class for model, including Part and Assembly,
    which should overwrite get_F_value() and get_smax()
    method. Then, both can be treated equally in SAS calculation.
    """
    def get_F_value(self, reciprocal_coord: Tensor) -> Tensor:
        """Get F value (scattering amplitude) of certain coordinates
        in reciprocal space.
        #* To be overwritten.

        Args:
            reciprocal_coord (Tensor): shape=(n, 3)

        Returns:
            Tensor: F value of corresponding coordinates
        """
        F_value = torch.ones(reciprocal_coord.shape[0], dtype=torch.complex64)
        return F_value

    def get_s_max(self) -> float:
        """Return maximum s value of a part or assembly.

        Returns:
            float: _description_
        """
        smax = 0.
        return smax


class Part(Model):
    """Parent class for part model.
    Subclass: StlPart and MathPart
    """
    def __init__(self, filename: str | None = None, partname: str | None = None, is_isolated: bool = True, device: str = 'cpu') -> None:
        """Init function of Part class

        Args:
            filename (str | None, optional): model file path, either .stl or .py file. Defaults to None.
            partname (str | None, optional): name of the part model, set to filename without extension if not given. Defaults to None.
            is_isolated (bool, optional): True for form factor only, False for structure factor. Defaults to True.
            device (str, optional): device of the model to be stored and calculated. Defaults to 'cpu'.
        """
        self.model_type = 'general'
        self.device = device
        self.partname = partname
        if filename is not None:
            self.filename = filename
            self.basename = os.path.basename(filename)
            if partname is None:
                self.partname = os.path.splitext(self.basename)[0]
        self.is_isolated = is_isolated
        self.geo_transform: list[dict] = []
        self.bound_min: tuple[float, float, float]
        self.bound_max: tuple[float, float, float]
        self.real_lattice_spacing: float
        self.real_lattice_1d_size: int
        self.reciprocal_lattice_1d_size: int
        self.x: Tensor
        self.y: Tensor
        self.z: Tensor
        self.x_original: Tensor
        self.y_original: Tensor
        self.z_original: Tensor
        self.sld: Tensor
        self.sld_original: Tensor
        self.centered: bool
        self.s1d: Tensor
        self.s1dz: Tensor
        self.F_half: Tensor
        self.sx: Tensor
        self.sy: Tensor
        self.sz_half: Tensor

    def _get_suggested_spacing(self, Lmin: float, Lmax:float, real_lattice_1d_size: int = 50) -> float:
        """Calculate optimal real_lattice_spacing and n_s values
        for the generation of reciprocal spacing. Based on my
        experience and test, the meshgrid density is set to be
        40 on Lmax. But must ensure that the lattice number in
        Lmin is lerger than 10. n_s the larger the better, but
        may use too much RAM or VRAM, speed is also slow.

        Args:
            Lmin (float): minimum real space scale of a model
            Lmax (float): maximum real space scale of a model
            real_lattice_1d_size (int, optional): number of points of real lattice in 1d. Defaults to 50.

        Returns:
            float: suggested spacing value
        """        
        n_d = real_lattice_1d_size
        real_lattice_spacing = Lmax / n_d
        real_lattice_spacing = min(real_lattice_spacing, Lmin/10)
        return real_lattice_spacing

    def set_real_lattice_meshgrid(self, x: Tensor, y: Tensor, z: Tensor) -> None:
        """Import lattice meshgrid from outside of class.
        For the need of some other function.
        Must be evenly spaced in all 3 dimensions.

        Args:
            x (Tensor): _description_
            y (Tensor): _description_
            z (Tensor): _description_
        """
        spacing = x[1,0,0].item() - x[0,0,0].item()
        self.real_lattice_spacing = spacing
        self.x, self.y, self.z = x.to(self.device), y.to(self.device), z.to(self.device)
        self.real_lattice_1d_size = max(x.shape)

    def gen_real_lattice_meshgrid(self, real_lattice_1d_size: int | None = None, spacing: float | None = None) -> tuple[Tensor, Tensor, Tensor]:
        """Generate equally spaced meshgrid in 3d real space.
        Can be assigned either real_lattice_1d_size or spacing.

        Args:
            real_lattice_1d_size (int | None, optional): _description_. Defaults to None.
            spacing (float | None, optional): _description_. Defaults to None.

        Returns:
            tuple[Tensor, Tensor, Tensor]: _description_
        """
        bound_min, bound_max = self.get_bound()
        L = tuple(map(lambda a, b: a-b, bound_max, bound_min))
        Lmin, Lmax = min(L), max(L)
        if real_lattice_1d_size is not None:
            spacing = self._get_suggested_spacing(Lmin, Lmax, real_lattice_1d_size=real_lattice_1d_size)
        elif spacing is None:
            spacing = self._get_suggested_spacing(Lmin, Lmax)
        
        lattice_min_1d = tuple(map(lambda bmin: bmin-spacing/2, bound_min))
        num_1d = tuple(map(lambda lmin, bmax: int((bmax-lmin)/spacing)+2, lattice_min_1d, bound_max))
        lattice_max_1d = tuple(map(lambda lmin, num: lmin+num*spacing, lattice_min_1d, num_1d))
        x1d, y1d, z1d = tuple(map(
            lambda lmin, lmax, num: torch.linspace(lmin, lmax, num, device=self.device),
            lattice_min_1d,
            lattice_max_1d,
            num_1d
        ))
        
        x, y, z = torch.meshgrid(x1d, y1d, z1d, indexing='ij')
        self.real_lattice_1d_size = max(num_1d)
        self.real_lattice_spacing = spacing
        self.x, self.y, self.z = x, y, z
        return x, y, z

    def set_real_lattice_sld(self, x: Tensor, y: Tensor, z: Tensor, sld: Tensor) -> None:
        """To directly import a real lattice with sld values, jump over StlPart or MathPart,
        Mainly for test use.

        Args:
            x (Tensor): _description_
            y (Tensor): _description_
            z (Tensor): _description_
            sld (Tensor): _description_
        """
        self.real_lattice_spacing = x[1,0,0].item() - x[0,0,0].item()
        self.x, self.y, self.z = x.to(self.device), y.to(self.device), z.to(self.device)
        self.sld = sld.to(self.device)
        self._store_original_real_lattice(x, y, z, sld)

    def get_bound(self) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        """Get boundary of part model. Return 2 points which
        determine a cuboid fully containing the whole part model.
        #* To be overwritten.

        Returns:
            tuple[tuple[float, float, float], tuple[float, float, float]]: _description_
        """
        xmin, ymin, zmin = self.x.min().item(), self.y.min().item(), self.z.min().item()
        xmax, ymax, zmax = self.x.max().item(), self.y.max().item(), self.z.max().item()
        bound_min, bound_max = (xmin, ymin, zmin), (xmax, ymax, zmax)
        self.bound_min, self.bound_max = bound_min, bound_max
        return bound_min, bound_max

    def gen_real_lattice_sld(self) -> Tensor:
        """Generate SLD lattice of this part model.
        Values at lattice points are corresponding SLD.
        Store original real lattice at last.
        #* To be overwritten.

        Returns:
            Tensor: sld values
        """
        x, y, z, sld = self.x, self.y, self.z, self.sld
        self._store_original_real_lattice(x, y, z, sld)
        return sld

    def _store_original_real_lattice(self, x: Tensor, y: Tensor, z: Tensor, sld: Tensor) -> None:
        """Store original real lattice in case of restore them

        Args:
            x (Tensor): _description_
            y (Tensor): _description_
            z (Tensor): _description_
            sld (Tensor): _description_
        """
        self.x_original = x.clone()
        self.y_original = y.clone()
        self.z_original = z.clone()
        self.sld_original = sld.clone()


    @log
    def gen_reciprocal_lattice(self, reciprocal_lattice_1d_size: int | None = None, need_centering: bool = True) -> Tensor:
        """Generate reciprocal lattice from real lattice.
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

        Args:
            reciprocal_lattice_1d_size (int | None, optional): _description_. Defaults to None.
            need_centering (bool, optional): _description_. Defaults to True.

        Returns:
            Tensor: scattering amplitude (F value)
        """
        # determine n_s in reciprocal space
        bound_min, bound_max = self.get_bound()
        xmin, ymin, zmin = bound_min
        xmax, ymax, zmax = bound_max
        if reciprocal_lattice_1d_size is None:
            if self.is_isolated:
                reciprocal_lattice_1d_size = 10 * self.real_lattice_1d_size
                reciprocal_lattice_1d_size = min(600, reciprocal_lattice_1d_size)  # in case of using too much resource
            else:
                reciprocal_lattice_1d_size = self.real_lattice_1d_size
        else:
            reciprocal_lattice_1d_size = max(reciprocal_lattice_1d_size, self.real_lattice_1d_size)
        n_s = reciprocal_lattice_1d_size
        # print('n_s: {}'.format(n_s))
        # larger n_s get more precise in low q,
        # but may use too many memory since F increases in ^3

        # size at z (3rd dim) is different with x & y (dim1&2)
        s1d = torch.fft.fftfreq(n_s, d=self.real_lattice_spacing, device=self.device)
        s1d = torch.fft.fftshift(s1d)
        s1dz = torch.fft.rfftfreq(n_s, d=self.real_lattice_spacing, device=self.device)

        # using rfft to save time. so only upper half (qz>=0)
        F_half = calcfunc.rfft3d(self.sld, n_s)

        # shift center to (0, 0, 0)
        # time-consuming part
        if need_centering:
            @log
            def centering(*args): # only for log use
                return self._translate_on_reciprocal_lattice(*args)
            xmin, ymin, zmin = xmin+self.real_lattice_spacing/2, ymin+self.real_lattice_spacing/2, zmin+self.real_lattice_spacing/2
            F_half = centering(F_half, s1d, s1d, s1dz, xmin, ymin, zmin)
            self.centered = True
        else:
            self.centered = False

        ##### Continuous-density correction #####
        # Correct discrete density to continuous density by multiplying
        # box scattering function from a voxel.
        # And slso eliminate the difference caused by different 
        # spacing in real space.
        # ATTENTION!
        #   Result from sphere shows applying continuous-density
        #   correction gives worse result then before, which shows
        #   larger deviation from -4 slope line.
        #
        # BEFORE: voxel volume correction
        # F_half = self.real_lattice_spacing**3 * F_half
        d = self.real_lattice_spacing
        sinc = lambda t: torch.nan_to_num(torch.sin(t)/t, nan=1.)
        qxd2, qyd2, qzd2 = 2*torch.pi*s1d*d/2, 2*torch.pi*s1d*d/2, 2*torch.pi*s1dz*d/2
        box_scatt = torch.einsum('i,j,k->ijk', sinc(qxd2), sinc(qyd2), sinc(qzd2))
        F_half = F_half * d**3 * box_scatt

        self.reciprocal_lattice_1d_size = reciprocal_lattice_1d_size
        self.s1d = s1d
        self.s1dz = s1dz
        self.F_half = F_half
        return F_half

    def _translate_on_reciprocal_lattice(self, F: Tensor, sx_1d: Tensor, sy_1d: Tensor, sz_1d: Tensor, vx: float, vy:float, vz:float) -> Tensor:
        """Default model center is at (0, 0, 0), referring to
        self.bound_min, self.bound_max. But real lattice bound
        for fft is (0, 0, 0) & self.bound_max-self.bound_min.
        So shift_vector = self.bound_min,
        shift_multiplier =  e^{-i * 2pi * s dot (-shif_vector)}.
        Shift_multiplier only change complex argument, so
        corresponding method is used here.
        Will generate reciprocal meshgrid sx, sy, sz.

        Args:
            F (Tensor): _description_
            sx_1d (Tensor): _description_
            sy_1d (Tensor): _description_
            sz_1d (Tensor): _description_
            vx (float): _description_
            vy (float): _description_
            vz (float): _description_

        Returns:
            Tensor: translated F values
        """
        sx, sy, sz_half = torch.meshgrid(sx_1d, sy_1d, sz_1d, indexing='ij')
        multiplier_arg = -1 * 2*torch.pi * (sx*vx + sy*vy + sz_half*vz)
        F_mod, F_arg = abi2modarg(F)
        F_new = modarg2abi(F_mod, F_arg+multiplier_arg)
        # will raise error when calculating exp on complex numbers on cuda
        self.sx, self.sy, self.sz_half = sx, sy, sz_half
        return F_new

    @log
    def translate(self, vx: float, vy: float, vz: float) -> None:
        """Translate model by vector.
        Change real space lattice directly;
        But in reciprocal space, change also depends on input
        coordinates. So transform methods like translate and
        rotate will return a closure function which will act
        on coordinates and reciprocal lattice in the later call.
        Method refer to self._translate_on_reciprocal_lattice()

        Args:
            vx (float): x component of translate vector
            vy (float): y component of translate vector
            vz (float): z component of translate vector

        Returns:
            no return
        """
        # real space
        def func_for_real(x, y, z, sld):
            return x+vx, y+vy, z+vz, sld

        # reciprocal
        v = torch.tensor([vx, vy, vz], dtype=torch.float32, device=self.device) # must claim dtype, or will raise error
        def func_for_reciprocal(reciprocal_coord: Tensor, multiplier_arg: Tensor) -> tuple[Tensor, Tensor]:
            """reciprocal_coord.shape == (n,3)
            multiplier_arg.shape = (n,)
            """
            arg = -2 * torch.pi * (reciprocal_coord @ v)  # only calculate argument of multiplier which is a complex number with mod=1
            return reciprocal_coord, arg+multiplier_arg

        self.geo_transform.append(dict(
            type = 'translate',
            param = (vx, vy, vz),
            func_for_real = func_for_real,  # 闭包函数
            func_for_reciprocal = func_for_reciprocal, # 闭包函数
        ))

    @log
    def rotate(self, axis: tuple[float, float, float], angle: float) -> None:
        """Rotate model around an axis passing origin by angle.
        Change real space lattice directly;
        But in reciprocal space, change also depends on input
        coordinates. So transform methods like translate and
        rotate will return a closure function which will act
        on coordinates and reciprocal lattice in the later call.
        Uses the Euler-Rodrigues formula. see
        https://en.wikipedia.org/wiki/Euler%E2%80%93Rodrigues_formula

        Args:
            axis (tuple[float, float, float]): vector describing direction of the axis
            angle (float): rotation angle in radians

        Returns:
            _type_: _description_
        """
        # real space
        def func_for_real(x, y, z, sld):
            points = torch.stack([x, y, z], dim=-1)
            points = euler_rodrigues_rotate(points, axis, angle)
            return *torch.unbind(points, dim=-1), sld

        # reciprocal
        def func_for_reciprocal(reciprocal_coord: Tensor, multiplier_arg: Tensor) -> tuple[Tensor, Tensor]:
            """_summary_

            Args:
                reciprocal_coord (Tensor): shape = (n,3)
                multiplier_arg (Tensor): shape = (n,)

            Returns:
                tuple[Tensor, Tensor]: _description_
            """
            new_reciprocal_coord = euler_rodrigues_rotate(reciprocal_coord, axis, -angle)
            return new_reciprocal_coord, multiplier_arg

        self.geo_transform.append(dict(
            type = 'rotate',
            param = (axis, angle),
            func_for_real = func_for_real,  # 闭包函数
            func_for_reciprocal = func_for_reciprocal, # 闭包函数
        ))

    def clear_geo_transform(self) -> None:
        """Clear all geometric transforms. Set part model to default.
        """
        self.x, self.y, self.z = self.x_original.clone(), self.y_original.clone(), self.z_original.clone()
        self.geo_transform = []

    @log
    def get_F_value(self, reciprocal_coord: Tensor, interpolation_method: Literal['trilinear', 'nearest'] = 'trilinear') -> Tensor:
        """Get F value (scattering amplitude) of certain coordinates
        in reciprocal space.

        Args:
            reciprocal_coord (Tensor): shape=(n, 3)

        Returns:
            Tensor: F value of corresponding coordinates
        """
        new_coord = reciprocal_coord.to(self.device)
        multiplier_arg = torch.zeros(reciprocal_coord.shape[0], device=self.device)
        # apply transform function for reciprocal
        if self.centered:
            for transform in reversed(self.geo_transform):
                new_coord, multiplier_arg = transform['func_for_reciprocal'](new_coord, multiplier_arg)

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
        if interpolation_method == 'trilinear':
            interp_func = calcfunc.trilinear_interp
        elif interpolation_method == 'nearest':
            interp_func = calcfunc.nearest_interp
        else:
            raise ValueError('Unsupported interpolation method: {}'.format(interpolation_method))
        F_value = interp_func(
            sx, sy, sz, self.s1d, self.s1d, self.s1dz, self.F_half, ds
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

        # apply translate
        F_value_mod, F_value_arg = abi2modarg(F_value)
        F_value = modarg2abi(F_value_mod, F_value_arg+multiplier_arg)

        output_device = reciprocal_coord.device
        return F_value.to(output_device)

    def get_s_max(self) -> float:
        """Return maximum s value available for a part model.

        Returns:
            float: maximum s value
        """
        smax = min(torch.abs(self.s1d.min()).item(), torch.abs(self.s1d.max()).item())
        return smax

    def get_real_lattice_sld(self, output_device: str = 'cpu') -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Return real lattice with sld values

        Args:
            output_device (str, optional): _description_. Defaults to 'cpu'.

        Returns:
            tuple[Tensor, Tensor, Tensor, Tensor]: real lattice with sld values
        """
        x = self.x.clone()
        y = self.y.clone()
        z = self.z.clone()
        sld = self.sld.clone()
        # apply transform
        for transform in self.geo_transform:
            x, y, z, sld = transform['func_for_real'](x, y, z, sld)
        x, y, z, sld = x.to(output_device), y.to(output_device), z.to(output_device), sld.to(output_device)
        return x, y, z, sld
    
    def get_sas(self, qx: Tensor, qy: Tensor, qz: Tensor, interpolation_method: Literal['trilinear', 'nearest'] = 'trilinear') -> Tensor:
        """Calculate SAS intensity pattern from reciprocal lattice
        according to given q coordinates (qx, qy, qz). No orientation
        average.
        Output dims of intensity and device will be same as input qx.
        #! Attention:
        #! The assumed q unit will still be the reverse of model
        #! unit. So be careful when generate q2d by detector geometry,
        #! should match that of model unit.

        Args:
            qx (Tensor): _description_
            qy (Tensor): _description_
            qz (Tensor): _description_
            interpolation_method (Literal[&#39;trilinear&#39;, &#39;nearest&#39;], optional): _description_. Defaults to 'trilinear'.

        Returns:
            Tensor: corresponding SAS values with same shape as input q values
        """
        smax = self.get_s_max()
        input_shape = qx.shape
        sx = qx.to(self.device).flatten()/(2*torch.pi)
        sy = qy.to(self.device).flatten()/(2*torch.pi)
        sz = qz.to(self.device).flatten()/(2*torch.pi)
        s = torch.sqrt(sx**2 + sy**2 + sz**2)
        sx[s>=smax], sy[s>=smax], sz[s>=smax] = 0., 0., 0.  # set value larger than smax to 0
        reciprocal_coord = torch.stack([sx, sy, sz], dim=-1)
        F = self.get_F_value(reciprocal_coord, interpolation_method=interpolation_method)
        F[s>=smax] = 0. # set value larger than smax to 0
        I = F.real**2 + F.imag**2
        I = I.reshape(input_shape)

        output_device = qx.device
        return I.to(output_device)

    def get_1d_sas(self, q1d: Tensor, orientation_average_offset: int = 100, interpolation_method: Literal['trilinear', 'nearest'] = 'trilinear') -> Tensor:
        """Calculate 1d SAS intensity curve from reciprocal lattice.
        Orientation averaged.
        Device of output tensor is the same as input q1d.
        #* The unit of q is assumed to be the reverse of model's length unit.

        Args:
            q1d (Tensor): 1d tensor
            orientation_average_offset (int, optional): _description_. Defaults to 100.
            interpolation_method (Literal[&#39;trilinear&#39;, &#39;nearest&#39;], optional): _description_. Defaults to 'trilinear'.

        Returns:
            Tensor: _description_
        """
        s_input = q1d.to(self.device)/(2*torch.pi)
        smax = self.get_s_max()
        s = s_input[torch.where(s_input<=smax)] # only calculate within available range

        # generate coordinates to interpolate using fibonacci grid
        # 每一个q值对应的球面取多少个取向进行平均
        n_on_sphere = s # s**2 increase too fast if use quadratic... time-consuming and unnecessary
        n_on_sphere = torch.round(n_on_sphere/n_on_sphere[0]) + orientation_average_offset
        sx, sy, sz = calcfunc.sampling_points(s, n_on_sphere)

        #### get value ####
        reciprocal_coord = torch.stack([sx, sy, sz], dim=-1)
        F = self.get_F_value(reciprocal_coord, interpolation_method=interpolation_method)
        I = F.real**2 + F.imag**2

        #### orientation average ####
        n_on_sphere = n_on_sphere.to('cpu')
        I_list = []
        begin_index = 0
        for N in n_on_sphere:
            N = int(N)
            Ii = torch.sum(I[begin_index:begin_index+N])/N
            I_list.append(Ii.to('cpu').item())
            begin_index += N

        #### output result ####
        output_device = q1d.device
        I = torch.tensor(I_list, device=output_device)
        I1d = -1 * torch.ones_like(s_input, device=output_device)
        I1d[s_input<=smax] = I
        return I1d
    
    def clone(self, deepcopy: bool = True) -> 'Part':
        """Clone self object. Use copy.deepcopy when deepcopy==True.
        But the self.geo_transform list will always be deepcopied.
        Use copy.copy() if absolute shallow copy needed.

        Args:
            deepcopy (bool, optional): _description_. Defaults to True.

        Returns:
            Part: _description_
        """
        if deepcopy:
            new_part = copy.deepcopy(self)
        else:
            new_part = copy.copy(self)
            new_part.geo_transform = copy.deepcopy(self.geo_transform)
        return new_part

    #========================================================
    #   functions for convinient and intuitionistic usage
    #========================================================
    @log
    def sample(self, real_lattice_1d_size: int | None = None, spacing: float | None = None) -> None:
        """To build the part model in real space.
        generate lattice model of a part model.

        Args:
            real_lattice_1d_size (int | None, optional): _description_. Defaults to None.
            spacing (float | None, optional): _description_. Defaults to None.
        """
        self.gen_real_lattice_meshgrid(real_lattice_1d_size=real_lattice_1d_size, spacing=spacing)
        self.gen_real_lattice_sld()

    @log
    def scatter(self, reciprocal_lattice_1d_size: int | None = None, need_centering: bool = True) -> None:
        """Simulate the scattering process to generate full reciprocal lattice.

        Args:
            reciprocal_lattice_1d_size (int | None, optional): _description_. Defaults to None.
        """
        self.gen_reciprocal_lattice(reciprocal_lattice_1d_size=reciprocal_lattice_1d_size, need_centering=need_centering)

    @log
    def measure(self, *qi: Tensor, orientation_average_offset: int = 100, interpolation_method: Literal['trilinear', 'nearest'] = 'trilinear') -> Tensor:
        """Simulate measurement process. The full scatter pattern is generated
        in self.scatter() process, this method gives measured result of certain
        q, return scattering intensity like real SAS measurements.
        Input should be either 1d q, which will return orientation averaged 1d I;
        or qx, qy, qz, which will return I(qx, qy, qz) with same shape.
        #! Attention:
        #! The assumed q unit will still be the reverse of model
        #! unit. So be careful when generate q2d by detector geometry,
        #! should match that of model unit.

        Args:
            orientation_average_offset (int, optional): _description_. Defaults to 100.
            interpolation_method (Literal[&#39;trilinear&#39;, &#39;nearest&#39;], optional): _description_. Defaults to 'trilinear'.

        Returns:
            Tensor: _description_
        """
        if len(qi) == 1:
            I = self.get_1d_sas(*qi, orientation_average_offset=orientation_average_offset, interpolation_method=interpolation_method)
        else:
            I = self.get_sas(*qi, interpolation_method=interpolation_method)
        return I
    
    def __call__(self, *args, **kwargs) -> Tensor:
        """Same as self.measure(), for convinience.
        """
        return self.measure(*args, **kwargs)


class StlPart(Part):
    """class for part from stl file.
    Rewrite get_bound and gen_real_lattice_sld methods.
    """
    def __init__(self, filename: str | None = None, partname: str | None = None, is_isolated: bool = True, sld_value: float = 1.0, mesh_centering : bool = True, device: str = 'cpu') -> None:
        """Load mesh from stl file

        Args:
            sld_value (float, optional): sld value of the whole model. Defaults to 1.0.
            mesh_centering (bool, optional): whether to center stl mesh to origin. Defaults to True.
            other args same as Part.__init__()
        """
        super().__init__(filename=filename, partname=partname, is_isolated=is_isolated, device=device)
        self.model_type = 'stl'
        if filename is not None:
            self.mesh = mesh.Mesh.from_file(self.filename)
            self.bound_min, self.bound_max = self.get_bound()
            if mesh_centering:# move model center to (0,0,0)
                center = self.mesh.get_mass_properties()[1]
                self.mesh.translate(-center)
        self.sld_value = sld_value

    def get_bound(self) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        """Get boundary of part model. Return 2 points which
        determine a cuboid fully containing the whole part model.

        Returns:
            tuple[tuple[float, float, float], tuple[float, float, float]]: min point and max point
        """
        vec = self.mesh.vectors
        vec = vec.reshape((vec.shape[0]*vec.shape[1], vec.shape[2]))
        bound_min = vec.min(axis=0)
        bound_max = vec.max(axis=0)
        bound_min, bound_max = tuple(bound_min.tolist()), tuple(bound_max.tolist())
        self.bound_min, self.bound_max = bound_min, bound_max
        return bound_min, bound_max

    @log
    def gen_real_lattice_sld(self) -> Tensor:
        """Generate SLD lattice of this stl part model.
        All same sld value inside stl part.
        Store original real lattice at last.

        Returns:
            Tensor: _description_
        """
        x, y, z = self.x, self.y, self.z
        origins = torch.stack(
            [x.flatten(), y.flatten(), z.flatten()],
            dim=1
        )
        ray = torch.rand(3) - 0.5
        ray = ray.to(self.device)
        vectors = self.mesh.vectors.copy()
        triangles = torch.from_numpy(vectors).to(torch.float32).to(self.device)
        intersect_count = calcfunc.moller_trumbore_intersect_count(origins, ray, triangles)
        index = intersect_count % 2   # 1 is in, 0 is out
        sld = self.sld_value * index
        sld = sld.reshape(x.size())

        self.sld = sld
        self._store_original_real_lattice(x, y, z, sld)
        return sld


class MathPart(Part):
    """class for part from math description.
    Rewrite get_bound and gen_real_lattice_sld methods.
    """
    def __init__(self, filename: str | None = None, math_model_class: type | None = None, partname: str | None = None, is_isolated: bool = True, device: str = 'cpu') -> None:        
        """Load math object from a specific math model class
        or py file. Provide either math_model_class or filename.

        Args:
            math_model_class (type | None, optional): direct import a math model class, child class of MathModelClassBase. Defaults to None.
            other args same as Part.__init__()
        """
        super().__init__(filename=filename, partname=partname, is_isolated=is_isolated, device=device)
        self.model_type = 'math'
        if math_model_class is not None:
            self.math_model: MathModelClassBase = math_model_class()
            try:
                self.partname = self.math_model.name
            except:
                pass
        elif filename is not None:
            abspath = os.path.abspath(filename)
            dirname = os.path.dirname(abspath)
            sys.path.append(dirname)
            stripped_name = os.path.splitext(self.basename)[0]
            module = __import__(stripped_name)
            self.math_model = module.MathModelClass()

    def set_params(self, **kwargs) -> None:
        """Set params in mathpart, change default param values.

        Args:
            keyword args: param_name = param_value
        """
        self.math_model.params.update(kwargs)
        
    def get_params(self) -> dict:
        """get a copy of current params dict

        Returns:
            dict: params dict
        """        
        return self.math_model.params.copy()

    def get_bound(self) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        """Get boundary of part model. Return 2 points which
        determine a cuboid fully containing the whole part model.

        Returns:
            tuple[tuple[float, float, float], tuple[float, float, float]]: min point and max point
        """
        bound_min, bound_max = self.math_model.get_bound()
        bound_min, bound_max = tuple(bound_min), tuple(bound_max)
        self.bound_min, self.bound_max = bound_min, bound_max
        return bound_min, bound_max

    @log
    def gen_real_lattice_sld(self) -> Tensor:
        """Generate sld in real lattice, which is sld value on each
        lattice meshgrid point. From math description.

        Returns:
            Tensor: sld values
        """
        x, y, z = self.x, self.y, self.z
        part_coord = self.math_model.coord
        u, v, w = convert_coord(x, y, z, 'car', part_coord)
        sld = self.math_model.sld(u, v, w) # will deal with device in function automatically
        if isinstance(sld, np.ndarray):
            sld = torch.from_numpy(sld).to(torch.float32).to(self.device)

        self.sld = sld
        self._store_original_real_lattice(x, y, z, sld)
        return sld
    
    
class PdbPart(Part):
    def __init__(self, filename: str | None = None, partname: str | None = None, probe: Literal['xray', 'neutron'] = 'xray', wavelength: float = 1.54, is_isolated: bool = True, device: str = 'cpu') -> None:
        super().__init__(filename=filename, partname=partname, is_isolated=is_isolated, device=device)
        self.model_type = 'pdb'
        self.probe = probe
        self.wavelength = wavelength
        self.read_pdb_structure()
        
    @log
    def read_pdb_structure(self):
        pdbparser = Bio.PDB.PDBParser(QUIET=True)   # suppress PDBConstructionWarning
        self.pdb_structure = pdbparser.get_structure(self.partname, self.filename)
        f, coord_list, covalent_radius = [], [], []
        if self.probe == 'neutron':
            atom_f_func = lambda pt_element: pt_element.neutron.b_c
        else:
            atom_f_func = lambda pt_element: pt_element.xray.scattering_factors(wavelength=self.wavelength)[0]
        for atom in self.pdb_structure.get_atoms():
            element = atom.element[0].upper() + atom.element[1:].lower()
            pt_element = pt.elements.symbol(element)
            f.append(atom_f_func(pt_element))
            coord_list.append(atom.coord)
            covalent_radius.append(pt_element.covalent_radius)
        self.atom_f = torch.tensor(f, dtype=torch.float32)
        self.atom_coord = torch.from_numpy(np.stack(coord_list, axis=0))
        self.atom_x, self.atom_y, self.atom_z = self.atom_coord[:,0], self.atom_coord[:,1], self.atom_coord[:,2]
        self.atom_covalent_radius = torch.tensor(covalent_radius)
    
    def get_bound(self) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        """Get boundary of part model. Return 2 points which
        determine a cuboid fully containing the whole part model.

        Returns:
            tuple[tuple[float, float, float], tuple[float, float, float]]: min point and max point
        """
        bound_min = tuple((self.atom_coord.min(dim=0).values).tolist())
        bound_max = tuple((self.atom_coord.max(dim=0).values).tolist())
        self.bound_min, self.bound_max = bound_min, bound_max
        return bound_min, bound_max
    
    @log
    def gen_real_lattice_sld(self) -> Tensor:
        x, y, z = self.x, self.y, self.z
        lattice_min = torch.tensor((x.min(), y.min(), z.min()))
        index = (self.atom_coord - lattice_min) / self.real_lattice_spacing
        index = index.round().to(torch.int64)
        sld = torch.zeros_like(x)
        sld[index[:,0], index[:,1], index[:,2]] = self.atom_f
        
        self.sld = sld
        self._store_original_real_lattice(x, y, z, sld)
        return sld


class Assembly(Part):
    """Assembly of several part model.
    For SAS calculation, use get_F_value for each part model.
    The own real lattice is for sld volume plot by using less points.
    """
    def __init__(self, *part: Part, device: str = 'cpu') -> None:
        """Initiate assembly model by several part models. Part models can be in different devices.

        Args:
            device (str, optional): device for assembly model, independent of any part model's device. Defaults to 'cpu'.
        """
        self.model_type = 'assembly'
        self.parts: list[Part] = []
        self.add_part(*part)
        self.device = device
        self.geo_transform = []  # should always be empty

    def add_part(self, *part: Part) -> None:
        """Add part object or list of part object to self.parts dictionary
        """
        for p in part:
            self.parts.append(p)

    @log
    def get_F_value(self, reciprocal_coord: Tensor, interpolation_method: Literal['trilinear', 'nearest'] = 'trilinear') -> Tensor:
        """Get F value (scattering amplitude) of certain coordinates
        in reciprocal space.

        Args:
            reciprocal_coord (Tensor): shape=(n, 3)
            interpolation_method (Literal[&#39;trilinear&#39;, &#39;nearest&#39;], optional): _description_. Defaults to 'trilinear'.

        Returns:
            Tensor: F value of corresponding coordinates
        """
        new_coord = reciprocal_coord.to(self.device)
        F_value = torch.zeros(new_coord.shape[0], dtype=torch.complex64, device=self.device)
        for part in self.parts:
            F_value = F_value + part.get_F_value(new_coord, interpolation_method=interpolation_method)
        output_device = reciprocal_coord.device
        return F_value.to(output_device)

    def get_s_max(self) -> float:
        """Maximum s value of assembly model.

        Returns:
            float: s_max value.
        """
        smax_list = []
        for part in self.parts:
            smax_list.append(part.get_s_max())
        return min(smax_list)
    
    def clear_geo_transform(self) -> None:
        for part in self.parts:
            part.clear_geo_transform()
    

    #========================================================
    #  generate own real lattice, only for sld volume plot
    #========================================================
    def get_bound(self) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        """Get maximum boundary containing all the parts.
        It is after all parts' transform

        Returns:
            tuple[tuple[float, float, float], tuple[float, float, float]]: _description_
        """
        l_xmin, l_ymin, l_zmin, l_xmax, l_ymax, l_zmax = [], [], [], [], [], []
        for part in self.parts:
            spacing = part.real_lattice_spacing
            x, y, z, _ = part.get_real_lattice_sld()
            l_xmin.append(x.min().item()-spacing/2)
            l_ymin.append(y.min().item()-spacing/2)
            l_zmin.append(z.min().item()-spacing/2)
            l_xmax.append(x.max().item()+spacing/2)
            l_ymax.append(y.max().item()+spacing/2)
            l_zmax.append(z.max().item()+spacing/2)
        xmin, ymin, zmin = min(l_xmin), min(l_ymin), min(l_zmin)
        xmax, ymax, zmax = max(l_xmax), max(l_ymax), max(l_zmax)
        bound_min, bound_max = (xmin, ymin, zmin), (xmax, ymax, zmax)
        return bound_min, bound_max

    def gen_real_lattice_sld(self) -> Tensor:
        """Generate lattice sld of assembly model's own.
        For sld volume plot use only, will not be used in sas calculation.
        Call self.gen_real_lattice_meshgrid() first.

        Returns:
            Tensor: _description_
        """
        sld = torch.zeros_like(self.x, dtype=torch.float32, device=self.device)
        for part in self.parts:
            #* restore to untransformed
            x, y, z = self.x.clone(), self.y.clone(), self.z.clone()
            for transform in reversed(part.geo_transform):
                if transform['type'] == 'translate':
                    vx, vy, vz = transform['param']
                    x, y, z = x-vx, y-vy, z-vz
                elif transform['type'] == 'rotate':
                    axis, angle = transform['param']
                    points = torch.stack([x, y, z], dim=-1)
                    points = euler_rodrigues_rotate(points, axis, -angle)
                    x, y, z = torch.unbind(points, dim=-1)
            temp_part = copy.deepcopy(part)
            temp_part.set_real_lattice_meshgrid(x, y, z)
            sld = sld + temp_part.gen_real_lattice_sld().to(self.device) # part models may be at different devices
        self.sld = sld
        return sld
