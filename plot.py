'''All kinds of plot, from 1d to 3d.
'''

import torch
from torch import Tensor
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import mpl_toolkits.mplot3d as mp3d
import functools

from model import Part, Assembly
from detector import Detector

def fig_ax_process(fig_kwargs: dict = {}, ax_kwargs: dict = {}):
    def fig_ax_process_decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            fig = kwargs.pop('fig', None)
            ax = kwargs.pop('ax', None)
            if ax is None:
                if fig is None:
                    fig = plt.figure(**fig_kwargs)
                ax = fig.add_subplot(**ax_kwargs)
            kwargs.update(dict(fig=fig, ax=ax))

            func(*args, **kwargs)

            savename = kwargs.pop('savename', None)
            show = kwargs.pop('show', True)
            if fig is not None:
                fig.tight_layout()
                if savename is not None:
                    fig.savefig(savename)
            if show:
                plt.show()
        return wrapper
    return fig_ax_process_decorator


def find_length(t: Tensor) -> float:
    return t.max().item() - t.min().item()

def find_center(t: Tensor) -> float:
    return (t.min().item() + t.max().item()) / 2

def process_equal_all_scale_range(x: Tensor, y: Tensor, z: Tensor) -> tuple[tuple, tuple, tuple]:
    lenx, leny, lenz = find_length(x), find_length(y), find_length(z)
    len_all = max(lenx, leny, lenz)
    cx, cy, cz = find_center(x), find_center(y), find_center(z)
    return (cx-len_all/2, cx+len_all/2), (cy-len_all/2, cy+len_all/2), (cz-len_all/2, cz+len_all/2)

def process_equal_xz_scale_range(x: Tensor, y: Tensor, z: Tensor) -> tuple[tuple, tuple, tuple]:
    lenx, lenz = find_length(x), find_length(z)
    len_xz = max(lenx, lenz)
    cx, cz = find_center(x), find_center(z)
    return (cx-len_xz/2, cx+len_xz/2), (y.min().item(), y.max().item()), (cz-len_xz/2, cz+len_xz/2)


@fig_ax_process(ax_kwargs=dict(projection='3d'))
def plot_parts(
    *parts: Part,
    fig: Figure | None = None,
    ax: Axes | None = None,
    show: bool = True,
    savename: str | None = None,
    ) -> None:
    '''Plot parts lattice in scatter plot.
    '''
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

    scale_range = process_equal_all_scale_range(x, y, z)
    ax.auto_scale_xyz(*scale_range)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

@fig_ax_process(ax_kwargs=dict(projection='3d'))
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

    scale_range = process_equal_all_scale_range(x, y, z)
    ax.auto_scale_xyz(*scale_range)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')


@fig_ax_process()
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
    q, I = q.to('cpu'), I.to('cpu')
    ax.plot(q, I, **kwargs)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'Q ($\mathrm{unit^{-1}}$)')
    ax.set_ylabel('Intensity (a.u.)')


@fig_ax_process()
def plot_sas2d(
    I2d: Tensor,
    do_log: bool = True,
    cmap: str = 'viridis',
    fig: Figure | None = None,
    ax: Axes | None = None,
    show: bool = True,
    savename: str | None = None,
    **kwargs
    ):
    '''Plot 1d SAS curve
    '''
    if do_log:
        I2d = torch.log(I2d)
    ax.imshow(I2d.T, origin='lower', cmap=cmap, **kwargs)


@fig_ax_process(ax_kwargs=dict(projection='3d', box_aspect=(1,2,1)))
def plot_real_space_detector(
    *dets: Detector,
    values: list[Tensor] | None = None,
    aspect: str = 'equal_xz',  # equal_all | equal_xz
    fig: Figure | None = None,
    ax: Axes | None = None,
    show: bool = True,
    savename: str | None = None
    ) -> None:
    '''Plot detector size and position in 3d.
    Can plot multiple detectors together. If only plot
    detector position, set values to None. If some of
    patterns are to be plot, then set the cooresponding
    value in values, and set others to None or other
    non-Tensor type. But len(values) == len(dets)
    '''
    # plot detector screen
    if values is not None:
        for det, value in zip(dets, values):
            if isinstance(value, Tensor):
                ax.plot_surface(det.x, det.y, det.z, facecolors=plt.cm.viridis(value/value.max()))
            else:
                ax.plot_surface(det.x, det.y, det.z)
    else:
        for det in dets:
            ax.plot_surface(det.x, det.y, det.z)

    # plot light edges on detector
    for det in dets:
        v0 = torch.tensor([0., 0., 0.])
        v1 = torch.tensor((det.x[0,0], det.y[0,0], det.z[0,0]))
        v2 = torch.tensor((det.x[0,-1], det.y[0,-1], det.z[0,-1]))
        v3 = torch.tensor((det.x[-1,-1], det.y[-1,-1], det.z[-1,-1]))
        v4 = torch.tensor((det.x[-1,0], det.y[-1,0], det.z[-1,0]))
        vtx_list = [
            torch.stack([v0,v1,v2], dim=0),
            torch.stack([v0,v2,v3], dim=0),
            torch.stack([v0,v3,v4], dim=0),
            torch.stack([v0,v4,v1], dim=0),
        ]
        tri = mp3d.art3d.Poly3DCollection(vtx_list)
        tri.set_color([(0.5, 0.5, 0.5, 0.2)])
        ax.add_collection3d(tri)

    # plot sample position at origin
    ax.scatter(0, 0, 0, color='k') # origin

    # plot direct x-ray
    l = []
    for det in dets:
        head = det.get_center()
        l.append(torch.abs(head[1]).item())
    arrow_length = max(l)
    ax.plot([0, 0], [0, arrow_length], [0, 0], color='k', linewidth=2)

    # scale plot to better illustrate
    lx, ly, lz = [0.], [0.], [0.]
    for det in dets:
        lx += [det.x.min(), det.x.max()]
        ly += [det.y.min(), det.y.max()]
        lz += [det.z.min(), det.z.max()]
    if 'all' in aspect:
        scale_range = process_equal_all_scale_range(torch.tensor(lx), torch.tensor(ly), torch.tensor(lz))
    else:
        scale_range = process_equal_xz_scale_range(torch.tensor(lx), torch.tensor(ly), torch.tensor(lz))
    ax.auto_scale_xyz(*scale_range)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

