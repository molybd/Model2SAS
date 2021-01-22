# -*- coding: UTF-8 -*-

import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from cycler import cycler

def genRgb(i, colormap='tab10', discrete_colormap=True):
    colors = eval('plt.cm.{}.colors'.format(colormap))
    N = len(colors)
    i = int(i)
    return colors[i%N]

def plotStlMeshes(mesh_list, label_list=None, show=True, figure=None):
    if label_list:
        use_legend = True
    else:
        use_legend = False
        label_list = [''] * len(mesh_list)

    # Create a new plot
    if figure:
        pass
    else:
        figure = plt.figure()
    axes = mplot3d.Axes3D(figure)

    # add the vectors to the plot
    temp = []  # for scale use
    for i in range(len(mesh_list)):
        mesh = mesh_list[i]
        temp.append(mesh.points.flatten())
        # plot model frame mesh
        Line3DCollection = mplot3d.art3d.Line3DCollection(
            mesh.vectors,
            linewidth=0.5,
            color=genRgb(i),
            label=label_list[i]
            )
        axes.add_collection3d(Line3DCollection)
    scale = np.hstack(temp)
    # Auto scale to the mesh size
    #scale = mesh.points.flatten()
    axes.auto_scale_xyz(scale, scale, scale)

    if use_legend:
        plt.legend()
    if show:
        plt.show()
    return figure
        

def plotPointsWithSld(points_with_sld, colormap='viridis', show=True, figure=None):
    # Create a new plot
    if figure:
        pass
    else:
        figure = plt.figure()
    axes = mplot3d.Axes3D(figure)

    x, y, z, sld = points_with_sld[:, 0], points_with_sld[:, 1], points_with_sld[:, 2], points_with_sld[:, 3]
    x, y, z, sld = x.flatten(), y.flatten(), z.flatten(), sld.flatten()
    c = sld - sld.min()  # make all positive values
    cm = plt.get_cmap(colormap)
    sc = axes.scatter(x, y, z, c=c, cmap=cm)
    plt.colorbar(sc)

    scale = points_with_sld[:,:-1].flatten()
    axes.auto_scale_xyz(scale, scale, scale)

    if show:
        plt.show()
    return figure

