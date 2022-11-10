'''All kinds of plot, from 1d to 3d.
'''

import torch
from torch import Tensor
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from model_new import Part, Assembly

def plot_parts(
    *parts: Part,
    fig: Figure | None = None,
    ax: Axes | None = None,
    show: bool = True,
    savename: str | None = None,
    ) -> None:
    '''Plot parts lattice in scatter plot.
    '''
    if ax is None:
        if fig is None:
            fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
    lx, ly, lz = [], [], []
    for part in parts:
        x, y, z, sld = part.get_real_lattice(output_device='cpu')
        x = x[torch.where(sld!=0)]
        y = y[torch.where(sld!=0)]
        z = z[torch.where(sld!=0)]
        ax.scatter(x, y, z)
        lx.append(x)
        ly.append(y)
        lz.append(z)
    x = torch.concat(lx)
    y = torch.concat(ly)
    z = torch.concat(lz)

    find_length = lambda t: t.max().item() - t.min().item()
    find_center = lambda t: (t.min().item() + t.max().item()) / 2
    lenx, leny, lenz = find_length(x), find_length(y), find_length(z)
    len_all = max(lenx, leny, lenz)
    cx, cy, cz = find_center(x), find_center(y), find_center(z)
    ax.auto_scale_xyz([cx-len_all/2, cx+len_all/2], [cy-len_all/2, cy+len_all/2], [cz-len_all/2, cz+len_all/2])

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    if fig is not None:
        fig.tight_layout()
        if savename is not None:
            fig.savefig(savename)
    if show:
        plt.show()


def plot_assembly(
    assembly: Assembly,
    cmap: str = 'viridis',
    fig: Figure | None = None,
    ax: Axes | None = None,
    show: bool = True,
    savename: str | None = None,
    ) -> None:
    '''Plot parts lattice in scatter plot.
    '''
    if ax is None:
        if fig is None:
            fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
    lx, ly, lz, lc = [], [], [], []
    
    for part in assembly.parts.values():
        x, y, z, sld = part.get_real_lattice(output_device='cpu')
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
    ax.scatter(x, y, z, c=c, cmap=cmap)

    find_length = lambda t: t.max().item() - t.min().item()
    find_center = lambda t: (t.min().item() + t.max().item()) / 2
    lenx, leny, lenz = find_length(x), find_length(y), find_length(z)
    len_all = max(lenx, leny, lenz)
    cx, cy, cz = find_center(x), find_center(y), find_center(z)
    ax.auto_scale_xyz([cx-len_all/2, cx+len_all/2], [cy-len_all/2, cy+len_all/2], [cz-len_all/2, cz+len_all/2])

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    if fig is not None:
        fig.tight_layout()
        if savename is not None:
            fig.savefig(savename)
    if show:
        plt.show()


def plot_sas1d(
    q: Tensor,
    I: Tensor,
    fig: Figure | None = None,
    ax: Axes | None = None,
    show: bool = True,
    savename: str | None = None,
    **kwargs
    ):
    '''Plot 1d SAS curve
    '''
    if ax is None:
        if fig is None:
            fig = plt.figure()
        ax = fig.add_subplot()

    q, I = q.to('cpu'), I.to('cpu')
    ax.plot(q, I, **kwargs)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'Q ($\mathrm{unit^{-1}}$)')
    ax.set_ylabel('Intensity (a.u.)')

    if fig is not None:
        fig.tight_layout()
        if savename is not None:
            fig.savefig(savename)
    if show:
        plt.show()