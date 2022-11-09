'''Generation of 1d SAS curve and
2d SAS pattern.
'''

import torch
from torch import Tensor

import calc_func_new as calc_func
from utility_new import timer
from model_new import Part, Assembly


class Sas:
    '''Class to generate 1d ort 2d SAS data
    '''
    def __init__(self, model: Part | Assembly) -> None:
        self.device = model.device
        self.model = model

    @timer
    def calc_sas1d(self, q1d: Tensor, orientation_average_offset: int = 100) -> tuple[Tensor, Tensor]:
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
        I1d = torch.tensor(I_list)

        return 2*torch.pi*s.to('cpu'), I1d


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from model_new import StlPart, MathPart

    def plot_parts(*parts):
        lx, ly, lz, lc = [], [], [], []
        for part in parts:
            x, y, z, sld = part.get_sld_lattice()
            x = x[torch.where(sld!=0)]
            y = y[torch.where(sld!=0)]
            z = z[torch.where(sld!=0)]
            sld = sld[torch.where(sld!=0)]
            lx.append(x)
            ly.append(y)
            lz.append(z)
            lc.append(sld)
        x = torch.concat(lx)
        y = torch.concat(ly)
        z = torch.concat(lz)
        c = torch.concat(lc)
        figure = plt.figure()
        ax = figure.add_subplot(projection='3d')
        ax.scatter(x, y, z, c=c)

    @timer
    def main():
        part_list = []
        for i in range(5):
            part1 = MathPart(filename=r'models/cylinder_x.py', device='cuda:1')
            part1.math_description.params = {
                'R': 10,
                'H': 50,
                'sld_value': 1
            }
            part1.gen_lattice_meshgrid()
            part1.gen_sld_lattice()
            part1.gen_reciprocal_lattice()
            part1.translate((i*50,0,0))
            part_list.append(part1)

        assembly = Assembly(*part_list)
        scatt = Sas(assembly)
        # q = torch.linspace(0.0001, 2, 200)
        q = torch.logspace(-4, 0.3, 200)
        q1, I1 = scatt.calc_sas1d(q)

        # plot_parts(*part_list)
        # plt.show()

        part3 = MathPart(filename=r'models/cylinder_x.py', device='cuda:0')
        part3.math_description.params = {
            'R': 10,
            'H': 250,
            'sld_value': 1
        }
        part3.gen_lattice_meshgrid(spacing=1)
        part3.gen_sld_lattice()
        part3.gen_reciprocal_lattice()
        # part3.rotate((0,0,1), torch.pi/3)

        scatt = Sas(part3)
        q2, I2 = scatt.calc_sas1d(q)

        # plot_parts(part3)
        # plt.show()

        plt.plot(q1, I1)
        plt.plot(q2, I2)
        plt.xscale('log')
        plt.yscale('log')
        plt.savefig('test.png')
        plt.show()
        
        
    main()