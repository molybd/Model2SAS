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
        input_shape = q2d_x.shape
        sx = q2d_x.to(self.device).flatten()/(2*torch.pi)
        sy = q2d_y.to(self.device).flatten()/(2*torch.pi)
        sz = q2d_z.to(self.device).flatten()/(2*torch.pi)
        reciprocal_coord = torch.stack([sx, sy, sz], dim=-1)
        F = self.model.get_F_value(reciprocal_coord)
        I = F.real**2 + F.imag**2
        I = I.reshape(input_shape)

        output_device = q2d_x.device
        return I.to(output_device)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from model import StlPart, MathPart
    from plot import plot_parts, plot_assembly, plot_sas1d

    def do_all(part, spacing=None, n_s=None, need_centering=True):
        part.gen_real_lattice_meshgrid(spacing=spacing)
        part.gen_real_lattice_sld()
        part.gen_reciprocal_lattice(n_s=n_s, need_centering=need_centering)

    @timer
    def main():
        # '''
        part_list = []
        for i in range(5):
            part1 = MathPart(filename=r'models/cylinder_x.py', device='cuda')
            part1.math_description.params = {
                'R': 10,
                'H': 50,
                'sld_value': 1
            }
            do_all(part1, spacing=1)
            part1.translate((i*50,0,0))
            part_list.append(part1)
        # plot_parts(*part_list, show=True)
        assembly = Assembly(*part_list)
        # plot_assembly(assembly, show=True)
        scatt = Sas(assembly)
        # q = torch.linspace(0.0001, 2, 200)
        q = torch.logspace(-4, 0.3, 200)
        I1 = scatt.calc_sas1d(q)

        # plot_parts(*part_list)
        # plt.show()

        part2 = MathPart(filename=r'models/cylinder_x.py', device='cpu')
        part2.math_description.params = {
            'R': 10,
            'H': 250,
            'sld_value': 1
        }
        do_all(part2, spacing=1, need_centering=False)
        scatt = Sas(part2)
        I2 = scatt.calc_sas1d(q)

        fig = plt.figure()
        ax = fig.add_subplot()
        plot_sas1d(q, I1, ax=ax, show=False)
        plot_sas1d(q, I2, ax=ax, show=False)
        plt.show()
        fig.savefig('test.png')
        # '''

        # part = StlPart(filename=r'models/torus.stl', device='cuda')
        # part = MathPart(filename=r'models/cylinder_y.py', device='cuda')
        # part.gen_real_lattice_meshgrid()
        # part.gen_real_lattice_sld()
        # part.gen_reciprocal_lattice()

        # part.rotate((1,0,0), torch.pi/2)
        # part.translate((0,100,200))

        # plot_parts(part, savename='test.png', show=True)

        # scatt = Sas(part)
        # q = torch.linspace(0.005, 2.5, 200)
        # q = torch.linspace(0.0001, 2, 200)
        # q, I = scatt.calc_sas1d(q)
        # plot_sas1d(q, I)

    def main1():
        part1 = MathPart(filename=r'models/cylinder_z.py', device='cuda:0')
        part1.math_description.params = {
                'R': 10,
                'H': 30,
                'sld_value': 1
            }
        part2 = StlPart(filename=r'models/torus.stl', device='cuda:1', sld_value=2)
        do_all(part1)
        do_all(part2)
        assembly = Assembly(part1, part2, device='cuda:1')

        plot_parts(part1, part2, savename='test1.png')
        plot_assembly(assembly, savename='test2.png')

        scatt = Sas(assembly)
        q = torch.linspace(0.001, 2, 200)
        # q = torch.logspace(-3, 0.3, 200)
        I1 = scatt.calc_sas1d(q)

        fig = plt.figure()
        ax = fig.add_subplot()
        plot_sas1d(q, I1, ax=ax, show=False)
        plt.show()
        fig.savefig('test3.png')
        
    def main2():
        from detector import Detector
        from std import AgBh, bull_tendon
        
        scatt = Sas(AgBh)
        # scatt = Sas(bull_tendon)

        det = Detector((981, 1043), 172e-6)
        det.set_sdd(1.5)
        det.translate(20e-3, 30e-3)
        qx, qy, qz = det.get_reciprocal_coord(1.342)
        I = scatt.calc_sas2d(qx, qy, qz)
        plt.imshow(torch.log(I.T))
        # plt.imshow(I.T)
        plt.show()
        plt.savefig('test.png')

        # q = torch.linspace(0.01, 1, 200)
        # I = scatt.calc_sas1d(q)
        # plt.plot(q, I)
        # plt.show()
        # plt.savefig('test.png')


    main2()