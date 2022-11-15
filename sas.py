'''Generation of 1d SAS curve and
2d SAS pattern.
'''

import torch
from torch import Tensor

import calc_func as calc_func
from utility import timer
from model import Part, Assembly


class Sas:
    '''Class to generate 1d ort 2d SAS data
    '''
    def __init__(self, model: Part | Assembly) -> None:
        self.device = model.device
        self.model = model

    @timer
    def calc_sas1d(self, q1d: Tensor, orientation_average_offset: int = 100, output_device: str = 'cpu') -> tuple[Tensor, Tensor]:
        '''Calculate 1d SAS curve from reciprocal lattice
        '''
        q1d = q1d.to(self.device)
        s_input = q1d/(2*torch.pi)
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
        I1d = torch.tensor(I_list, device=output_device)

        return 2*torch.pi*s.to(output_device), I1d


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
        q1, I1 = scatt.calc_sas1d(q)

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
        q2, I2 = scatt.calc_sas1d(q)

        fig = plt.figure()
        ax = fig.add_subplot()
        plot_sas1d(q1, I1, ax=ax, show=False)
        plot_sas1d(q2, I2, ax=ax, show=False)
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
        q1, I1 = scatt.calc_sas1d(q)

        fig = plt.figure()
        ax = fig.add_subplot()
        plot_sas1d(q1, I1, ax=ax, show=False)
        plt.show()
        fig.savefig('test3.png')
        
        
    main1()