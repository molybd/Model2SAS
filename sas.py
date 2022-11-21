'''Generation of 1d SAS curve and
2d SAS pattern.
'''

import torch
from torch import Tensor

import calc_func as calc_func
from utility import timer
from model import Part, Assembly, Model


class Sas:
    '''Class to generate 1d ort 2d SAS data
    '''
    def __init__(self, model: Part | Assembly | Model) -> None:
        self.device = model.device
        self.model = model

    @timer
    def calc_sas1d(self, q1d: Tensor, orientation_average_offset: int = 100) -> Tensor:
        '''Calculate 1d SAS intensity curve from reciprocal lattice.
        Orientation averaged.
        Device of output tensor is the same as input q1d.
        '''
        s_input = q1d.to(self.device)/(2*torch.pi)
        smax = self.model.get_s_max()
        s = s_input[torch.where(s_input<=smax)]

        # generate coordinates to interpolate using fibonacci grid
        # 每一个q值对应的球面取多少个取向进行平均
        n_on_sphere = s#**2 # increase too fast if use quadratic... time-consuming and unnecessary
        n_on_sphere = torch.round(n_on_sphere/n_on_sphere[0]) + orientation_average_offset
        sx, sy, sz = calc_func.sampling_points(s, n_on_sphere)

        #### interpolate
        reciprocal_coord = torch.stack([sx, sy, sz], dim=-1)
        F = self.model.get_F_value(reciprocal_coord)
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

        output_device = q1d.device
        I = torch.tensor(I_list, device=output_device)
        I1d = -1 * torch.ones_like(s_input, device=output_device)
        I1d[s_input<=smax] = I
        return I1d

    def calc_sas2d(self, q2d_x: Tensor, q2d_y: Tensor, q2d_z: Tensor) -> Tensor:
        '''Calculate 2d SAS intensity pattern from reciprocal lattice
        according to given q coordinates (qx, qy, qz). dims of q2d_x,
        q2d_y, q2d_z is 2.
        Device of output tensor is the same as input q2d.
        Attention:
        The assumed q unit will still be the reverse of model
        unit. So be careful when generate q2d by detector geometry,
        should match that of model unit.
        '''
        smax = self.model.get_s_max()
        input_shape = q2d_x.shape
        sx = q2d_x.to(self.device).flatten()/(2*torch.pi)
        sy = q2d_y.to(self.device).flatten()/(2*torch.pi)
        sz = q2d_z.to(self.device).flatten()/(2*torch.pi)
        s = torch.sqrt(sx**2 + sy**2 + sz**2)
        sx[s>=smax], sy[s>=smax], sz[s>=smax] = 0., 0., 0.  # set value larger than smax to 0
        reciprocal_coord = torch.stack([sx, sy, sz], dim=-1)
        F = self.model.get_F_value(reciprocal_coord)
        F[s>=smax] = 0. # set value larger than smax to 0
        I = F.real**2 + F.imag**2
        I = I.reshape(input_shape)

        output_device = q2d_x.device
        return I.to(output_device)
