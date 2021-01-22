# -*- coding: UTF-8 -*-

import os
import numpy as np
from stl import mesh
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from shutil import copyfile

from ModelSection import stlmodel
from Functions import intensity, xyz2sph


class model2sas:
    ''' A project that contain model and calculation

    Attributes:
    model: model object
    data: data object
    '''

    def __init__(self, dir):
        ''' 创建project文件夹dir，并且把工作目录设在此文件夹下
        这样后面就可以都使用相对位置
        '''
        self.dir = os.path.abspath(dir)
        try:
            os.mkdir(dir)
        except FileExistsError:
            print('dir already exists')
        finally:
            os.chdir(dir)
            print('project dir: {}'.format(dir))

    def genModel(self, modelname=None):
        self.model = model(modelname)

    def genData(self):
        self.data = data(self.model.points_with_sld)

    def importFile(self, filepath, sld):
        basename = os.path.basename(filepath)
        destination_path = os.path.join(self.dir, basename)
        try:
            copyfile(filepath, destination_path)
        except SameFileError as e:
            print(str(e))

        filetype = filepath.split('.')[-1].lower()
        if filetype == 'stl':
            self.model.importStlFile(filepath, sld)
        elif filetype == 'py':
            pass



class model:
    ''' a 3d model
    A points model from stl file (.stl) or math description (.py) or saved numpy txt (.txt) file

    Attributes:

    Methods:

    '''

    def __init__(self, modelname=None):
        # filename is a relative path
        self.modelname = modelname    # 如果是None，等到后面有文件输入后以文件名做modelname
        self.stlmodel_list = []

    def importStlFile(self, filepath, sld):
        filepath = os.path.abspath(filepath)
        sld = float(sld)
        this_stl_model = stlmodel(filepath, sld)
        self.stlmodel_list.append(this_stl_model)

    def genPoints(self, interval=None, grid_num=10000):
        '''Generate points model from configured several models
        In case of translating or rotating model sections, importing file part
        and generating points model parts are separated.
        So please configure your model before this step!
        '''
        # determine the overall boundary first
        stlmodel_list = self.stlmodel_list
        min_boundary_points_list, max_boundary_points_list = [], []
        for stlmodel in stlmodel_list:
            min_point, max_point = stlmodel.getBoundaryPoints()
            min_boundary_points_list.append(min_point)
            max_boundary_points_list.append(max_point)
        min_boundary_points = np.vstack(min_boundary_points_list)
        max_boundary_points = np.vstack(max_boundary_points_list)
        boundary_min = np.min(min_boundary_points, axis=0)
        boundary_max = np.max(max_boundary_points, axis=0)

        # determine interval
        if interval:
            interval = interval
        else:
            scale = boundary_max - boundary_min
            # grid_num defauld is 10000
            interval = (scale[0]*scale[1]*scale[2] / grid_num)**(1/3)

        # generate grid
        grid = self._genGrid(boundary_min, boundary_max, interval)   # shape == (n, 3)
        
        # calculate in model index for each stlmodel
        for stlmodel in stlmodel_list:
            stlmodel.importGrid(grid)
            stlmodel.calcInModelGridIndex()

        # combine all the stl model
        # !! ATTENTION !!
        # I choose to use the higher sld value for the overlapped point
        sld_grid_index_list = []
        for stlmodel in stlmodel_list:
            sld_grid_index_list.append(stlmodel.sld_grid_index)
        sld_grid_index_stack = np.vstack(sld_grid_index_list)
        sld_grid_index = np.max(sld_grid_index_stack, axis=0)

        points = grid[np.where(sld_grid_index!=0)]
        slds = sld_grid_index[np.where(sld_grid_index!=0)]
        slds = slds.reshape((slds.size,1))
        points_with_sld = np.hstack((points, slds))


        self.grid = grid
        self.interval = interval
        self.sld_grid_index = sld_grid_index
        self.stlmodel_list = stlmodel_list
        self.points = points
        self.points_with_sld = points_with_sld # shape==(n, 4) 前三列是坐标，最后一列是相应的sld

    def importMathFile(self):
        pass

    def _genGrid(self, boundary_min, boundary_max, interval):
        '''Generate grid points
        boundary_min = np.array([xmin, ymin, zmin])
        boundary_max = np.array([xmax, ymax, zmax])
        '''
        xmin, ymin, zmin = boundary_min[0], boundary_min[1], boundary_min[2]
        xmax, ymax, zmax = boundary_max[0], boundary_max[1], boundary_max[2]
        xscale = np.linspace(xmin, xmax, num=int((xmax-xmin)/interval+1))
        yscale = np.linspace(ymin, ymax, num=int((ymax-ymin)/interval+1))
        zscale = np.linspace(zmin, zmax, num=int((zmax-zmin)/interval+1))
        x, y, z = np.meshgrid(xscale, yscale, zscale)
        x, y, z = x.reshape(x.size,1), y.reshape(y.size,1), z.reshape(z.size,1)
        grid = np.hstack((x, y, z))
        return grid   # shape == (n, 3)


class data:

    def __init__(self, points_with_sld):
        self.points_with_sld = points_with_sld
        self.points = points_with_sld[:,:3]
        self.slds = points_with_sld[:,-1]

    def genQ(self, qmin, qmax, qnum=200, logq=False):
        if logq:
            q = np.logspace(np.log10(qmin), np.log10(qmax), num=qnum, base=10, dtype='float32')
        else:
            q = np.linspace(qmin, qmax, num=qnum, dtype='float32')
        return q

    def calcSas(self, q, lmax=50):
        points = self.points
        slds = self.slds
        I = intensity(q, points, slds, lmax)

        self.q = q
        self.I = I
        self.lmax = lmax


##### 测试使用 #####
def plotPoints(points):
    fig = plt.figure()
    axes = mplot3d.Axes3D(fig)

    axes.scatter(points[:,0], points[:,1], points[:,2], color='k')
    # Show the plot to the screen
    plt.show()

def plotSection(stlmodel_list):
    fig = plt.figure()
    axes = mplot3d.Axes3D(fig)

    for stlmodel in stlmodel_list:
        axes.scatter(stlmodel.points[:,0], stlmodel.points[:,1], stlmodel.points[:,2])
    # Show the plot to the screen
    plt.show()
################


if __name__ == "__main__":
    model = model()
    model.importStlFile('models\\shell_12_large_hole.STL', 1)
    #model.importStlFile('models\\torus.STL', 2)
    #model.importStlFile('models\\torus.STL', 4)
    # 平移与旋转模型
    #model.stlmodel_list[1].mesh.translate([30,10,10])
    #model.stlmodel_list[1].mesh.rotate([1,0,0], theta=np.pi/2, point=[0,0,0])
    
    model.genPoints()

    
    #plotSection(model.stlmodel_list)

    #plotPoints(model.points)

    model_data = data(model.points_with_sld)
    q = model_data.genQ(0.001, 1)
    model_data.calcSas(q, lmax=50)

    np.savetxt('test.dat', np.vstack((q, model_data.I)).T)

    plt.plot(q, model_data.I)
    plt.xscale('log')
    plt.yscale('log')
    plt.show()

