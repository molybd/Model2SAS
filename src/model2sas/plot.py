"""All kinds of plot, from 1d to 3d.
"""

import functools
import os
from typing import Literal

import torch
from torch import Tensor

import plotly.graph_objects as go

from .model import Part, Assembly


#==================================
# Utility functions for plot
#==================================

def plot_utils(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        fig: go.Figure = func(*args, **kwargs)
        title = kwargs.pop('title', None)
        show = kwargs.pop('show', True)
        savename = kwargs.pop('savename', None)
        plotly_template = kwargs.pop('plotly_template', 'plotly')
        
        fig.update_layout(template=plotly_template)
        if title is not None:
            fig.update_layout(title_text=title)
        if show:
            fig.show()
        if savename is not None:
            if os.path.splitext(savename)[-1] == '.html':
                write_html(savename, fig.to_html(), encoding='utf-8')
            else:
                fig.write_image(savename)
        return fig
    return wrapper

def write_html(filename: str, htmlstr: str, encoding: str = 'utf-8') -> str:
    """Write html string to a html file.
    Reason of implementing this instead of using plotly.io.write_html()
    is that plotly method doesn't support encoding option, will causs
    error on Windows.

    Args:
        filename (str): _description_
        htmlstr (str): _description_
        encoding (str, optional): _description_. Defaults to 'utf-8'.

    Returns:
        str: _description_
    """
    with open(filename, 'w', encoding=encoding) as f:
        f.write(htmlstr)
        return os.path.abspath(filename)
    

class Voxel(go.Mesh3d):
    def __init__(self, xc=None, yc=None, zc=None, spacing=None, **kwargs):
        x, y, z, i, j, k = self.gen_vertices_triangles(xc, yc, zc, spacing)
        super().__init__(x=x, y=y, z=z, i=i, j=j, k=k, **kwargs)

    def gen_vertices_triangles(self, xc, yc, zc, spacing) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Generate vertices and triangles for mesh plot.
        For each point (xc, yc, zc), generate a cubic box with edge_length = spacing

        Args:
            xc: x coordinates of center point
            yc: y coordinates of center point
            zc: z coordinates of center point
            spacing: spacing of mesh grid, and cubic box edge length

        Returns:
            tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]: _description_
        """
        s = spacing
        xv = torch.stack((xc-s/2, xc+s/2, xc+s/2, xc-s/2, xc-s/2, xc+s/2, xc+s/2, xc-s/2), dim=1).flatten()
        yv = torch.stack((yc-s/2, yc-s/2, yc+s/2, yc+s/2, yc-s/2, yc-s/2, yc+s/2, yc+s/2), dim=1).flatten()
        zv = torch.stack((zc-s/2, zc-s/2, zc-s/2, zc-s/2, zc+s/2, zc+s/2, zc+s/2, zc+s/2), dim=1).flatten()

        i0 = torch.tensor((0, 2, 0, 5, 1, 6, 2, 7, 3, 4, 4, 6))
        j0 = torch.tensor((1, 3, 1, 4, 2, 5, 3, 6, 0, 7, 5, 7))
        k0 = torch.tensor((2, 0, 5, 0, 6, 1, 7, 2, 4, 3, 6, 4))
        seq = 8 * torch.arange(xc.numel())
        i = (torch.unsqueeze(i0, 0) + torch.unsqueeze(seq, 1)).flatten()
        j = (torch.unsqueeze(j0, 0) + torch.unsqueeze(seq, 1)).flatten()
        k = (torch.unsqueeze(k0, 0) + torch.unsqueeze(seq, 1)).flatten()
        return xv, yv, zv, i, j, k

#==================================
# Plot functions below
#==================================
@plot_utils
def plot_1d_sas(
    q: Tensor | list[Tensor],
    I: Tensor | list[Tensor],
    name: str | list[str] | None = None,
    mode: str | list[str] = 'lines+markers',
    logx: bool = True,
    logy: bool = True,
    title: str | None = None,
    show: bool = True,
    savename: str | None = None,
    plotly_template: str | dict = 'plotly',
    ) -> go.Figure:
    """plot 1d SAS curve(s).
    q, I, name combinations: 1q, 1I, 1name | 1q, I list, name list | q list, I list, name list
    name: can be None
    mode options: lines | markers | lines+markers, same as q,I,name, can be assigned to each trace by list

    Args:
        q (Tensor | list[Tensor]): _description_
        I (Tensor | list[Tensor]): _description_
        name (str | list[str] | None, optional): _description_. Defaults to None.
        mode (str | list[str], optional): lines | markers | lines+markers, same as q,I,name, can be assigned to each trace by list. Defaults to 'lines+markers'.
        logx (bool, optional): _description_. Defaults to True.
        logy (bool, optional): _description_. Defaults to True.
        title (str | None, optional): _description_. Defaults to None.
        show (bool, optional): _description_. Defaults to True.
        savename (str | None, optional): _description_. Defaults to None.

    Returns:
        go.Figure
    """
    def gen_list(x, length:int) -> list:
        if isinstance(x, list):
            x_list = x
        else:
            x_list = [x] * length
        return x_list

    if isinstance(I, list):
        I_list = I
        n = len(I_list)
        q_list, name_list, mode_list = gen_list(q, n), gen_list(name, n), gen_list(mode, n)
    else:
        q_list, I_list, name_list, mode_list = [q], [I], [name], [mode]

    fig = go.Figure()
    for qi, Ii, namei, modei in zip(q_list, I_list, name_list, mode_list):
        fig.add_trace(go.Scatter(
            x=qi, y=Ii,
            mode=modei, # 'lines' or 'markers' or 'lines+markers'
            name=namei
            ))

    fig.update_layout(xaxis_title=r'Q (1/unit)')
    fig.update_layout(yaxis_title=r'Intensity (a.u.)')
    if logx:
        fig.update_xaxes(type='log')
    if logy:
        fig.update_yaxes(type='log')
    return fig


@plot_utils
def plot_2d_sas(
    I2d: Tensor,
    logI: bool = True,
    title: str | None = None,
    colorscale: str | None = None,
    show: bool = True,
    savename: str | None = None,
    plotly_template: str | dict = 'plotly',
    ) -> go.Figure:
    """Plot a 2d SAS pattern.

    Args:
        I2d (Tensor): _description_
        logI (bool, optional): _description_. Defaults to True.
        title (str | None, optional): _description_. Defaults to None.
        colorscale (str | None, optional): _description_. Defaults to None.
        show (bool, optional): _description_. Defaults to True.
        savename (str | None, optional): _description_. Defaults to None.

    Returns:
        go.Figure
    """
    fig = go.Figure()
    if logI:
        data = torch.log10(I2d)
        data = torch.nan_to_num(data, nan=0., neginf=0.) # incase 0 in data, cause log(0) output
    else:
        data = I2d
    fig.add_trace(go.Heatmap(
        z=data.T,
        colorscale=colorscale
        ))
    fig.update_xaxes(
        scaleanchor='y',
        scaleratio=1,
        constrain='domain'
        )
    return fig


@plot_utils
def plot_3d_sas(
    qx: Tensor,
    qy: Tensor,
    qz: Tensor,
    I3d: Tensor,
    logI: bool = True,
    title: str | None = None,
    colorscale: str | None = None,
    show: bool = True,
    savename: str | None = None,
    plotly_template: str | dict = 'plotly',
    ) -> go.Figure:
    """_summary_

    Args:
        qx (Tensor): _description_
        qy (Tensor): _description_
        qz (Tensor): _description_
        I3d (Tensor): _description_
        logI (bool, optional): _description_. Defaults to True.
        title (str | None, optional): _description_. Defaults to None.
        colorscale (str | None, optional): _description_. Defaults to None.
        show (bool, optional): _description_. Defaults to True.
        savename (str | None, optional): _description_. Defaults to None.
        plotly_template (str | dict, optional): _description_. Defaults to 'plotly'.

    Returns:
        go.Figure: _description_
    """    
    fig = go.Figure()
    if logI:
        data = torch.log10(I3d)
        data = torch.nan_to_num(data, nan=0., neginf=0.) # incase 0 in data, cause log(0) output
    else:
        data = I3d
    fig.add_trace(go.Volume(
        x=qx.flatten(),
        y=qy.flatten(),
        z=qz.flatten(),
        value=data.flatten(),
        opacity=0.1,
        surface_count=21,
        coloraxis='coloraxis'
    ))
    fig.update_layout(scene_aspectmode='data') # make equal aspect
    fig.update_layout(coloraxis={'colorscale': colorscale})
    return fig


@plot_utils
def plot_model(
    *model: Part | Assembly,
    type: Literal['volume', 'voxel'] | None = None,
    title: str | None = None,
    colorscale: str = 'Plasma',
    show: bool = True,
    savename: str | None = None,
    plotly_template: str | dict = 'plotly',
    ) -> go.Figure:
    """Plot models in 3d
    If type not specified, voxel for part model, volume for assembly model.
    Voxel plot is better to inspect shape without sld;
    Volume plot can see through model with sld distribution.

    Args:
        type (Literal[&#39;volume&#39;, &#39;voxel&#39;] | None, optional): type of plot. Defaults to None.
        title (str | None, optional): _description_. Defaults to None.
        colorscale (str, optional): _description_. Defaults to 'Plasma'.
        show (bool, optional): _description_. Defaults to True.
        savename (str | None, optional): _description_. Defaults to None.

    Raises:
        ValueError: _description_

    Returns:
        go.Figure
    """
    fig = go.Figure()
    for modeli in model:
        if isinstance(modeli, Assembly):
            modeli.sample()
            x, y, z, sld = modeli.get_real_lattice_sld(output_device='cpu')
            spacing = modeli.real_lattice_spacing
            name = 'assembly'
            if type is None:
                type = 'volume'
        else: # isinstance(modeli, Part)
            x, y, z, sld = modeli.get_real_lattice_sld(output_device='cpu')
            spacing = modeli.real_lattice_spacing
            name = modeli.partname
            if type is None:
                type = 'voxel'
        
        if type == 'voxel':
            fig.add_trace(Voxel(
                xc=x[sld!=0],
                yc=y[sld!=0],
                zc=z[sld!=0],
                spacing=spacing,
                name=name,
                showlegend=True
                ))
        elif type == 'volume':
            fig.add_trace(go.Volume(
                x=x.flatten(),
                y=y.flatten(),
                z=z.flatten(),
                value=sld.flatten(),
                opacity=0.1,
                surface_count=21,
                coloraxis='coloraxis'
            ))
        else:
            raise ValueError('Unsupported plot type: {}'.format(type))

    fig.update_layout(scene_aspectmode='data') # make equal aspect
    if type == 'volume':
        fig.update_layout(coloraxis={'colorscale': colorscale})
    return fig


@plot_utils
def plot_surface(
    *data: tuple[Tensor, ...],
    logI: bool = True,
    title: str | None = None,
    colorscale: str = 'Plasma',
    show: bool = True,
    savename: str | None = None,
    plotly_template: str | dict = 'plotly',
    ) -> go.Figure:
    """Plot a surface in real or reciprocal space by coordinates,
    such as detector surface in reciprocal space.

    Args:
        data (tuple[Tensor, ...]): (x, y, z) or (x, y, z, I2d), all Tensor should be 2d
        logI (bool, optional): _description_. Defaults to True.
        title (str | None, optional): _description_. Defaults to None.
        colorscale (str, optional): _description_. Defaults to 'Plasma'.
        show (bool, optional): _description_. Defaults to True.
        savename (str | None, optional): _description_. Defaults to None.

    Returns:
        go.Figure: _description_
    """
    if len(data[0]) == 4:
        value_provided = True
    else:
        value_provided = False
    
    fig = go.Figure()
    for i, datai in enumerate(data):
        if value_provided:
            x, y, z, I2d = datai
            if logI:
                surfacecolor = torch.log10(I2d)
                surfacecolor = torch.nan_to_num(surfacecolor, nan=0., neginf=0.) # incase 0 in data, cause log(0) output
            else:
                surfacecolor = I2d
        else:
            x, y, z = datai[:3]
            surfacecolor = i * torch.ones_like(x)

        fig.add_trace(go.Surface(
            x=x, y=y, z=z, surfacecolor=surfacecolor, coloraxis='coloraxis'
        ))
    fig.update_layout(coloraxis = {'colorscale': colorscale})
    
    fig.update_layout(scene_aspectmode='data') # make equal aspect, or use fig.update_scenes(aspectmode='data')
    return fig


@plot_utils
def plot_real_detector(
    *data: tuple[Tensor, ...],
    logI: bool = True,
    title: str | None = None,
    colorscale: str = 'Plasma',
    show: bool = True,
    savename: str | None = None,
    plotly_template: str | dict = 'plotly',
    ) -> go.Figure:
    """Plot detector(s) surface in realspace, also display direct beam
    and covered solid angle by detector

    Args:
        data (tuple[Tensor, ...]): (x, y, z) | (x, y, z, I2d), all Tensor should be 2d
        logI (bool, optional): _description_. Defaults to True.
        title (str | None, optional): _description_. Defaults to None.
        colorscale (str, optional): _description_. Defaults to 'Plasma'.
        show (bool, optional): _description_. Defaults to True.
        savename (str | None, optional): _description_. Defaults to None.

    Returns:
        go.Figure: _description_
    """
    # plot detector surface
    fig = plot_surface(*data, logI=logI, colorscale=colorscale, show=False)

    fig.add_trace(go.Scatter3d(
        x=[0,],
        y=[0,],
        z=[0,],
        mode='markers',
        showlegend=False,
    )) # sample position at origin
    end_point_y = max([datai[1].max().item() for datai in data])
    fig.add_trace(go.Scatter3d(
        x=[0, 0],
        y=[0, end_point_y],
        z=[0, 0],
        mode='lines',
        showlegend=False,
    )) # direct beam

    # plot light edges on detector
    for datai in data:
        x, y, z = datai[:3]
        v0 = torch.tensor([0., 0., 0.])
        v1 = torch.tensor((x[0,0], y[0,0], z[0,0]))
        v2 = torch.tensor((x[0,-1], y[0,-1], z[0,-1]))
        v3 = torch.tensor((x[-1,-1], y[-1,-1], z[-1,-1]))
        v4 = torch.tensor((x[-1,0], y[-1,0], z[-1,0]))
        x, y, z = torch.unbind(
            torch.stack([v0,v1,v2,v3,v4], dim=1),
            dim=0
            )
        fig.add_trace(go.Mesh3d(
            x=x,
            y=y,
            z=z,
            alphahull=0,
            color='gray',
            opacity=0.1,
        ))

    fig.update_layout(scene_aspectmode='data') # make equal aspect, or use fig.update_scenes(aspectmode='data')
    return fig
