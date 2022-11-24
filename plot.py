'''All kinds of plot, from 1d to 3d.
'''

import torch
from torch import Tensor
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import mpl_toolkits.mplot3d as mp3d
from mpl_toolkits.mplot3d.axes3d import Axes3D
import functools

from model import Part, Assembly
from detector import Detector


#==================================
# Utility functions for plot
#==================================

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

def voxel_border(x: Tensor, y: Tensor, z: Tensor):
    '''x, y, z are 3d tensor, coordinates of meshgrid
    voxel center.
    '''
    def expand_1d(t1d):
        spacing = t1d[1] - t1d[0]
        tnew = torch.zeros(t1d.shape[0] + 1)
        tnew[:-1] = t1d[:]
        tnew[-1] = t1d[-1] + spacing
        return tnew
    xb = expand_1d(x[:,0,0])
    yb = expand_1d(y[0,:,0])
    zb = expand_1d(z[0,0,:])
    return torch.meshgrid(xb, yb, zb, indexing='ij')

def norm(t: Tensor) -> Tensor:
    return (t-t.min()) / (t.max()-t.min())



#==================================
# Plot functions below
#==================================

@fig_ax_process(ax_kwargs=dict(projection='3d'))
def plot_parts(
    *parts: Part,
    type: str = 'scatter',  # 'scatter' or 'voxels'
    fig: Figure | None = None,
    ax: Axes3D | None = None,
    show: bool = True,
    savename: str | None = None,
    ) -> None:
    '''Plot parts lattice in scatter plot.
    '''
    for part in parts:
        x, y, z, sld = part.get_real_lattice(output_device='cpu')
        if 'vox' in type:
            xb, yb, zb = voxel_border(x, y, z)
            model_shape = sld != 0.
            ax.voxels(xb, yb, zb, model_shape)
        else:
            x = x[torch.where(sld!=0)]
            y = y[torch.where(sld!=0)]
            z = z[torch.where(sld!=0)]
            ax.scatter(x, y, z)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    ax.set_aspect('equal')

@fig_ax_process(ax_kwargs=dict(projection='3d'))
def plot_assembly(
    assembly: Assembly,
    type: str = 'scatter',  # 'scatter' or 'voxels'
    colormap: str = 'viridis',
    fig: Figure | None = None,
    ax: Axes3D | None = None,
    show: bool = True,
    savename: str | None = None,
    ) -> None:
    '''Plot parts lattice in scatter plot.
    '''
    if 'vox' in type:
        assembly.gen_real_lattice_meshgrid()
        assembly.gen_real_lattice_sld()
        x, y, z, sld = assembly.get_real_lattice()
        xb, yb, zb = voxel_border(x, y, z)
        model_shape = sld != 0.
        colors = mpl.colormaps[colormap](norm(sld))
        ax.voxels(xb, yb, zb, model_shape, facecolors=colors)
    else:
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
        ax.scatter(x, y, z, c=c, cmap=colormap)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    ax.set_aspect('equal')


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
    colormap: str = 'viridis',
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
    ax.imshow(I2d.T, origin='lower', cmap=colormap, **kwargs)


@fig_ax_process(ax_kwargs=dict(projection='3d'))
def plot_real_space_detector(
    *dets: Detector,
    values: list[Tensor] | None = None,
    aspect: str = 'equal_xz',  # equal_all | equal_xz
    colormap: str = 'viridis',
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
                value = torch.nan_to_num(value, nan=0., neginf=0.) # in case of -inf after log
                colors = mpl.colormaps[colormap](norm(value))
                ax.plot_surface(det.x, det.y, det.z, facecolors=colors)
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

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    if 'all' in aspect:
        ax.set_aspect('equal')
    else:
        ax.set_box_aspect((1,2,1))
        ax.set_aspect('auto')


@fig_ax_process(ax_kwargs=dict(projection='3d'))
def plot_reciprocal_space_detector(
    *coords: tuple[Tensor, Tensor, Tensor],
    values: list[Tensor] | None = None,
    colormap: str = 'viridis',
    fig: Figure | None = None,
    ax: Axes | None = None,
    show: bool = True,
    savename: str | None = None
    ) -> None:
    '''Plot detector plane in reciprocal space (q space).
    If only plot detector position, set values to None.
    If some of patterns are to be plot, then set the
    cooresponding value in values, and set others to None
    or other non-Tensor type. But len(values) == len(dets)
    '''
    if values is not None:
        for coord, value in zip(coords, values):
            qx, qy, qz = coord
            if isinstance(value, Tensor):
                value = torch.nan_to_num(value, nan=0., neginf=0.)
                colors = mpl.colormaps[colormap](norm(value))
                ax.plot_surface(qx, qy, qz, facecolors=colors)
            else:
                ax.plot_surface(qx, qy, qz)
    else:
        for coord in coords:
            qx, qy, qz = coord
            ax.plot_surface(qx, qy, qz)
    
    ax.scatter(0, 0, 0, color='k') # origin

    ax.set_xlabel('Qx')
    ax.set_ylabel('Qy')
    ax.set_zlabel('Qz')

    ax.set_aspect('equal')