import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

def plot_stl_meshes(figure:Figure, meshes:list, labels:list=None, show:bool=True):
    '''plot multiple stl meshes in one figure
    '''
    if labels:
        use_legend = True
    else:
        use_legend = False
        labels = [''] * len(meshes)

    ax = mplot3d.Axes3D(figure)

    temp = []  # for scale use
    for i, mesh in enumerate(meshes):
        line_3d_collection = mplot3d.art3d.Line3DCollection(
            mesh.vectors,
            linewidth=0.5,
            label=labels[i]
            )
        ax.add_collection3d(line_3d_collection)
        temp.append(mesh.points.flatten())
    scale = np.hstack(temp)
    # Auto scale to the mesh size
    #scale = mesh.points.flatten()
    ax.auto_scale_xyz(scale, scale, scale)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    if use_legend:
        ax.legend()  # 在GUI里绘制图像应当避免使用plt的方法而是使用面向对象的绘图方式，否则plt画出来的东西在GUI中不显示
    #figure.set_tight_layout(True)  # 三维图似乎与tight_layout()不兼容
    if show:
        plt.show()
    
