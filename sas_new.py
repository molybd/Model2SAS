'''Generation of 1d SAS curve and
2d SAS pattern.
'''

import torch
from torch import Tensor

import calc_func_new as calc_func
from utility_new import timer, convert_coord
from model_new import Assembly, Model


class Sas:
    '''Class to generate 1d ort 2d SAS data
    '''
    def __init__(self, model: Model) -> None:
        self.model = model

    @timer
    def calc_sas1d(self, q1d: Tensor, orientation_average_offset: int = 100) -> tuple[Tensor, Tensor]:
        '''Calculate 1d SAS curve from reciprocal lattice
        '''
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
        I = torch.real(F)**2 + torch.imag(F)**2

        # orientation average
        I_list = []
        begin_index = 0
        for N in n_on_sphere:
            N = int(N)
            Ii = torch.sum(I[begin_index:begin_index+N])/N
            I_list.append(Ii)
            begin_index += N
        I1d = torch.tensor(I_list)

        return 2*torch.pi*s, I1d


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from model_new import StlPart, MathPart

    def plot_parts(*parts):
        lx, ly, lz, lc = [], [], [], []
        for part in parts:
            x = part.x[torch.where(part.sld_lattice!=0)].flatten()
            y = part.y[torch.where(part.sld_lattice!=0)].flatten()
            z = part.z[torch.where(part.sld_lattice!=0)].flatten()
            c = part.sld_lattice[torch.where(part.sld_lattice!=0)].flatten()
            lx.append(x)
            ly.append(y)
            lz.append(z)
            lc.append(c)
        x = torch.concat(lx)
        y = torch.concat(ly)
        z = torch.concat(lz)
        c = torch.concat(lc)
        figure = plt.figure()
        ax = figure.add_subplot(projection='3d')
        ax.scatter(x, y, z, c=c)

    @timer
    def main():
        # part1 = StlPart(filename=r'models\torus.stl')
        part1 = MathPart(filename=r'models\spherical_core_shell.py')
        part1.math_description.params = {
            'R_core': 8,
            'thickness': 2,
            'sld_value': 1,
        }
        part1.gen_lattice_meshgrid()
        part1.gen_sld_lattice()
        part1.gen_reciprocal_lattice()

        part2 = MathPart(filename=r'models\spherical_core_shell.py')
        part2.math_description.params = {
            'R_core': 0,
            'thickness': 8,
            'sld_value': 1,
        }
        part2.gen_lattice_meshgrid()
        part2.gen_sld_lattice()
        part2.gen_reciprocal_lattice()

        assembly = Assembly(part1, part2)
        scatt = Sas(assembly)
        q = torch.linspace(0.001, 1, 200)
        q1, I1 = scatt.calc_sas1d(q)

        plot_parts(part1, part2)
        plt.show()

        part3 = MathPart(filename=r'models\spherical_core_shell.py')
        part3.math_description.params = {
            'R_core': 0,
            'thickness': 10,
            'sld_value': 1,
        }
        part3.gen_lattice_meshgrid()
        part3.gen_sld_lattice()
        part3.gen_reciprocal_lattice()

        scatt = Sas(part3)
        q = torch.linspace(0.001, 1, 200)
        q2, I2 = scatt.calc_sas1d(q)

        plot_parts(part3)
        plt.show()

        plt.plot(q1, I1)
        plt.plot(q2, I2)
        plt.xscale('log')
        plt.yscale('log')
        plt.show()
        
    main()