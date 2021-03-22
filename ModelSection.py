# -*- coding: UTF-8 -*-

import numpy as np
from stl import mesh
import os, sys

from Functions import coordConvert


class stlmodel:

    def __init__(self, filepath, sld):
        self.filepath = os.path.abspath(filepath)
        self.name = os.path.basename(filepath)
        self.sld = sld
        self.mesh = mesh.Mesh.from_file(filepath)

    def getBoundaryPoints(self):
        vectors = self.mesh.vectors
        xmin, xmax, ymin, ymax, zmin, zmax = np.min(vectors[:,:,0]), np.max(vectors[:,:,0]), np.min(vectors[:,:,1]), np.max(vectors[:,:,1]), np.min(vectors[:,:,2]), np.max(vectors[:,:,2])
        return np.array([xmin, ymin, zmin]), np.array([xmax, ymax, zmax])

    def importGrid(self, grid):
        self.grid = grid

    def calcInModelGridIndex(self):
        grid = self.grid
        vectors = self.mesh.vectors

        # determine whether points inside the model
        ray = np.random.rand(3) + 0.01     # in case that all coordinates are 0, which is almost impossible
        intersect_count = np.zeros(grid.shape[0])
        for triangle in vectors:
            intersect_count += self._isIntersect(grid, ray, triangle)
        in_model_grid_index = intersect_count % 2   # 1 is in, 0 is out
        sld_grid_index = self.sld * in_model_grid_index # the sld for each point in grid
        points = grid[np.where(in_model_grid_index != 0)] # screen points in model
        
        self.in_model_grid_index = in_model_grid_index
        self.sld_grid_index = sld_grid_index
        self.points = points
        return in_model_grid_index  # shape == (n,)


    def _isIntersect(self, origins, ray, triangle):
        '''Calculate all the points intersect with 1 triangle
        using Möller–Trumbore intersection algorithm
        see paper https://doi.org/10.1080/10867651.1997.10487468

        Args:
            origins: ndarray, shape == (n, 3)
            ray: ndarray, shape==(3,), direction of ray
            triangle: ndarray, shape==(3,3) vertices of a triangle, shape

        Returns:
            1darray, shape == (n,), 与输入的点(origins)一一对应, 如果与三角形有交集那么该点对应的值为1，否则为0
        '''
        O = origins
        D = ray
        V0 = triangle[0]
        V1 = triangle[1]
        V2 = triangle[2]
        E1 = V1 - V0
        E2 = V2 - V0
        T = O - V0
        P = np.cross(D, E2)
        Q = np.cross(T, E1)
        det = np.dot(P, E1)
        if abs(det) >= np.finfo(np.float32).eps: #因为三角形里使用的是float32，所以在float32下判断是否等于0
            tuv = (1/det) * np.vstack((np.dot(Q,E2), np.dot(T,P), np.dot(Q,D))).T
            ispositive = np.sign(tuv) + 1  #大于等于零的数会变成正数（1或2），小于零的数会变成 0，结果和原数组形状完全相同，一一对应
            isallpositive =  ispositive[:,0]*ispositive[:,1]*ispositive[:,2] #只要tuv三个中有一个数小于零那么这一行结果就是0
            uplusv = -1*np.sign(tuv[:,1] + tuv[:,2] - 1) + 1 #判断u+v是否小于1，小于1的那一行变成正数，大于1的那一行结果是0
            return np.sign(isallpositive * uplusv) #最终输出的结果是对应着输入每一个点的一维数组，如果与三角形有交集那么该点对应的值为1，否则为0
        else:
            return np.zeros(origins.shape[0])



class mathmodel:

    def __init__(self, filepath):
        self.filepath = os.path.abspath(filepath)
        self.name = os.path.basename(filepath)
        dirname, basename = os.path.split(filepath)
        sys.path.append(dirname)
        module_name = os.path.splitext(basename)[0]
        mathmodel_module = __import__(module_name)
        mathmodel_object = mathmodel_module.specific_mathmodel()
        self.specific_mathmodel = mathmodel_object
        self.genSamplePoints()

    def getBoundaryPoints(self):
        boundary_min = self.specific_mathmodel.boundary_min
        boundary_max = self.specific_mathmodel.boundary_max
        return boundary_min, boundary_max

    def importGrid(self, grid):
        self.grid = grid

    def calcInModelGridIndex(self):
        grid = self.grid
        specific_mathmodel = self.specific_mathmodel

        # change grid coords (xyz) to destination coords
        coord = specific_mathmodel.coord
        grid_in_coord = coordConvert(grid, 'xyz', coord)

        in_model_grid_index = specific_mathmodel.shape(grid_in_coord)
        sld_grid_index = specific_mathmodel.sld()
        points = grid[np.where(in_model_grid_index != 0)] # screen points in model

        self.in_model_grid_index = in_model_grid_index
        self.sld_grid_index = sld_grid_index
        self.points = points
        return in_model_grid_index  # shape == (n,)

    def genSamplePoints(self, interval=None, grid_num=10000):
        specific_mathmodel = self.specific_mathmodel
        # generate grid for sample points
        boundary_min = self.specific_mathmodel.boundary_min
        boundary_max = self.specific_mathmodel.boundary_max
        # determine interval for sample points
        if interval:
            interval = interval
        else:
            scale = boundary_max - boundary_min
            # grid_num defauld is 10000
            interval = (scale[0]*scale[1]*scale[2] / grid_num)**(1/3)

        xmin, ymin, zmin = boundary_min[0], boundary_min[1], boundary_min[2]
        xmax, ymax, zmax = boundary_max[0], boundary_max[1], boundary_max[2]
        xscale = np.linspace(xmin, xmax, num=int((xmax-xmin)/interval+1))
        yscale = np.linspace(ymin, ymax, num=int((ymax-ymin)/interval+1))
        zscale = np.linspace(zmin, zmax, num=int((zmax-zmin)/interval+1))
        x, y, z = np.meshgrid(xscale, yscale, zscale)
        x, y, z = x.reshape(x.size,1), y.reshape(y.size,1), z.reshape(z.size,1)
        grid = np.hstack((x, y, z))

        # change grid coords (xyz) to destination coords
        coord = specific_mathmodel.coord
        grid_in_coord = coordConvert(grid, 'xyz', coord)

        in_model_grid_index = specific_mathmodel.shape(grid_in_coord)
        sld_grid_index = specific_mathmodel.sld()
        points = grid[np.where(in_model_grid_index != 0)] # screen points in model
        sld = sld_grid_index[np.where(in_model_grid_index != 0)]  # 用 in_model_grid_index 的原因是有可能出现sld=0但是实际在模型内的点
        sld = sld.reshape((sld.size, 1))
        points_with_sld = np.hstack((points, sld))

        self.sample_points = points
        self.sample_points_with_sld = points_with_sld
        return points_with_sld

        


