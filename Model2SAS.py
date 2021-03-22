# -*- coding: UTF-8 -*-

import os
import numpy as np
from stl import mesh
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from shutil import copyfile

from ModelSection import stlmodel, mathmodel
from Functions import intensity, xyz2sph, intensity_parallel
from Plot import *


class model2sas:
    ''' A project that contain model and calculation

    Attributes:
    model: model object
    data: data object
    '''

    def __init__(self, name):
        ''' Make project folder: dir/name
        '''
        
        self.name = name
        self.setupModel()

    def setupModel(self):
        self.model = model(name=self.name)

    def importFile(self, filepath, sld=1):
        filepath = os.path.abspath(filepath)
        basename = os.path.basename(filepath)
        filetype = filepath.split('.')[-1].lower()
        if filetype == 'stl':
            self.model.importStlFile(filepath, sld)
        elif filetype == 'py':
            self.model.importMathFile(filepath)

    def genPoints(self, interval=None, grid_num=10000):
        self.model.genPoints(interval=interval, grid_num=grid_num)
        self.points_with_sld = self.model.points_with_sld

    def savePointsWithSld(self, filename):
        header = 'x\ty\tz\tsld'
        np.savetxt(filename, self.points_with_sld, header=header)

    def setupData(self):
        self.data = data(self.model.points_with_sld)

    def calcSas(self, qmin, qmax, qnum=200, logq=False, lmax=50, parallel=False, core_num=2, proc_num=4):
        q = self.data.genQ(qmin, qmax, qnum=qnum, logq=logq)
        self.data.calcSas(q, lmax=lmax, parallel=parallel, core_num=core_num, proc_num=proc_num)
        self.q = self.data.q
        self.I = self.data.I
        #self.saveSasData()

    def saveSasData(self, filename):
        header = 'q\tI\tpseudo error(I/1000)'
        data = np.vstack((self.data.q, self.data.I, self.data.error)).T
        np.savetxt(filename, data, header=header)




class model:
    ''' a 3d model
    A points model from stl file (.stl) or math description (.py) or saved numpy txt (.txt) file

    Attributes:

    Methods:

    '''

    def __init__(self, name='model'):
        # filename is a relative path
        self.name = name
        self.stlmodel_list = []
        self.mathmodel_list = []

    def importStlFile(self, filepath, sld):
        filepath = os.path.abspath(filepath)
        sld = float(sld)
        this_stlmodel = stlmodel(filepath, sld)
        self.stlmodel_list.append(this_stlmodel)
    
    def importMathFile(self, filepath):
        filepath = os.path.abspath(filepath)
        this_mathmodel = mathmodel(filepath)
        self.mathmodel_list.append(this_mathmodel)


    def genPoints(self, interval=None, grid_num=10000):
        '''Generate points model from configured several models
        In case of translating or rotating model sections, importing file part
        and generating points model parts are separated.
        So please configure your model before this step!

        Also, stl model and math model can be used in the same project.
        So in this method, points are generated for all the model section.
        '''
        # determine the overall boundary first
        stlmodel_list = self.stlmodel_list
        mathmodel_list = self.mathmodel_list
        min_boundary_points_list, max_boundary_points_list = [], []
        for stlmodel in stlmodel_list:
            min_point, max_point = stlmodel.getBoundaryPoints()
            min_boundary_points_list.append(min_point)
            max_boundary_points_list.append(max_point)
        for mathmodel in mathmodel_list:
            min_point, max_point = mathmodel.getBoundaryPoints()
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
        
        # calculate in model index for each stlmodel and mathmodel
        for stlmodel in stlmodel_list:
            stlmodel.importGrid(grid)
            stlmodel.calcInModelGridIndex()
        for mathmodel in mathmodel_list:
            mathmodel.importGrid(grid)
            mathmodel.calcInModelGridIndex()

        # combine all the model sections
        # !! ATTENTION !!
        # I choose to use the sum(sld) value for the overlapped point
        in_model_grid_index_list = []
        sld_grid_index_list = []
        for stlmodel in stlmodel_list:
            in_model_grid_index_list.append(stlmodel.in_model_grid_index)
            sld_grid_index_list.append(stlmodel.sld_grid_index)
        for mathmodel in mathmodel_list:
            in_model_grid_index_list.append(mathmodel.in_model_grid_index)
            sld_grid_index_list.append(mathmodel.sld_grid_index)
        in_model_grid_index_stack = np.vstack(in_model_grid_index_list)
        in_model_grid_index = np.sign(np.sum(in_model_grid_index_stack, axis=0))
        sld_grid_index_stack = np.vstack(sld_grid_index_list)
        sld_grid_index = np.sum(sld_grid_index_stack, axis=0)

        points = grid[np.where(in_model_grid_index!=0)]
        sld = sld_grid_index[np.where(in_model_grid_index!=0)]  # 用 in_model_grid_index 的原因是有可能出现sld=0但是实际在模型内的点
        sld = sld.reshape((sld.size,1))
        points_with_sld = np.hstack((points, sld))

        self.grid = grid
        self.interval = interval
        self.sld_grid_index = sld_grid_index
        self.stlmodel_list = stlmodel_list
        self.points = points
        self.points_with_sld = points_with_sld # shape==(n, 4) 前三列是坐标，最后一列是相应的sld

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
        self.sld = points_with_sld[:,-1]

    def genQ(self, qmin, qmax, qnum=200, logq=False):
        if logq:
            q = np.logspace(np.log10(qmin), np.log10(qmax), num=qnum, base=10, dtype='float32')
        else:
            q = np.linspace(qmin, qmax, num=qnum, dtype='float32')
        return q

    def calcSas(self, q, lmax=50, parallel=False, core_num=2, proc_num=4):
        points = self.points
        sld = self.sld
        if parallel:
            I = intensity_parallel(q, points, sld, lmax, core_num=core_num, proc_num=proc_num)
        else:
            I = intensity(q, points, sld, lmax)

        self.q = q
        self.I = I
        self.error = 0.001 * I   # 默认生成千分之一的误差，主要用于写文件的占位
        self.lmax = lmax


if __name__ == "__main__":
    project = model2sas('test')
    project.importFile('models\cylinder.py')
    #plotStlMeshes([stlmodel.mesh for stlmodel in project.model.stlmodel_list], label_list=[stlmodel.name for stlmodel in project.model.stlmodel_list])
    
    #plotPoints(project.model.mathmodel_list[0].sample_points)
    #plotPointsWithSld(project.model.mathmodel_list[0].sample_points_with_sld)
    
    project.genPoints(grid_num=100000)
    plotPointsWithSld(project.model.points_with_sld, figure=plt.figure())
    # np.savetxt('project_points_with_sld.txt', project.points_with_sld)
    '''
    project.setupData()
    project.calcSas(0.01, 1, parallel=True)
    plotSasCurve(project.q, project.I)
    '''

    