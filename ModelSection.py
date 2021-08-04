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
        #ray = np.random.rand(3) + 0.01     # in case that all coordinates are 0, which is almost impossible
        ray = np.array([1., 1., 1.])
        
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
            triangle: ndarray, shape==(3,3) vertices of a triangle

        Returns:
            1darray, shape == (n,), 与输入的点(origins)一一对应, 如果与三角形有交集那么该点对应的值为1，否则为0
        '''
        # 全部使用float32计算，速度快不少
        O = origins.astype(np.float32)
        D = ray.astype(np.float32)
        V0 = triangle[0].astype(np.float32)
        V1 = triangle[1].astype(np.float32)
        V2 = triangle[2].astype(np.float32)
        E1 = V1 - V0
        E2 = V2 - V0
        T = O - V0
        P = np.cross(D, E2)
        Q = np.cross(T, E1)
        det = np.dot(P, E1)
        if abs(det) >= np.finfo(np.float32).eps: #因为三角形里使用的是float32，所以在float32下判断是否等于0
            t, u, v = np.dot(Q,E2)/det, np.dot(T,P)/det, np.dot(Q,D)/det
            intersect = np.zeros_like(t)
            intersect[(t>0) & (u>0) & (v>0) & ((u+v)<1)] = 1
            return intersect
        else:
            return np.zeros(origins.shape[0])
    
    '''
    # 完全向量化的方法，计算速度并不比循环快，内存反倒用的更多，因此不使用这个方法
    def calcInModelGridIndex_vectorized(self, slice_num=None):
        grid = self.grid
        vectors = self.mesh.vectors

        # determine whether points inside the model
        #ray = np.random.rand(3) + 0.01     # in case that all coordinates are 0, which is almost impossible
        ray = np.array([1., 1., 1.])

        import psutil
        free_memory = psutil.virtual_memory().free
        if slice_num:
            slice_num = int(slice_num)
        else:
            max_size = grid.shape[0] * vectors.shape[0] * 3 * 2 * 4  # 使用float32，每个数字4byte
            slice_num = int(max_size/(0.9*free_memory)) + 1
        slice_length = grid.shape[0] // slice_num + 1
        count_list = []
        for i in range(slice_num):
            index_begin = i * slice_length
            index_end = (i+1) * slice_length
            intersect_count = self._countIntersect(grid[index_begin:index_end, :], ray, vectors)
            count_list.append(intersect_count)
        intersect_count = np.hstack(count_list)
        in_model_grid_index = intersect_count % 2   # 1 is in, 0 is out
        sld_grid_index = self.sld * in_model_grid_index # the sld for each point in grid
        points = grid[np.where(in_model_grid_index != 0)] # screen points in model
        
        self.in_model_grid_index = in_model_grid_index
        self.sld_grid_index = sld_grid_index
        self.points = points
        return in_model_grid_index  # shape == (n,)
        

    # 下面是完全数组化重写的结果，结果耗时差不多，而且会很占内存，效果不好
    def _countIntersect(self, origins, ray, triangles):
        \'''Calculate all the points intersect with all triangles
        using Möller–Trumbore intersection algorithm
        see paper https://doi.org/10.1080/10867651.1997.10487468

        Args:
            origins: ndarray, shape == (n, 3)
            ray: ndarray, shape==(3,), direction of ray
            triangles: ndarray, shape==(m, 3, 3), vertices of triangles

        Returns:
            1darray, shape == (n,), 与输入的点(origins)一一对应, 如果与三角形有交集那么该点对应的值为1，否则为0
        \'''
        n, m = origins.shape[0], triangles.shape[0]
        O = origins.astype(np.float32)  # (n, 3)
        D = ray.astype(np.float32)  # (3,)
        V0 = triangles[:, 0, :].astype(np.float32)  # (m, 3)
        V1 = triangles[:, 1, :].astype(np.float32)  # (m, 3)
        V2 = triangles[:, 2, :].astype(np.float32)  # (m, 3)
        E1 = V1 - V0  # (m, 3)
        E2 = V2 - V0  # (m, 3)
        
        O_ext = np.array([O]*m)  # (m, n, 3)
        O_ext = np.swapaxes(O_ext, 0, 1)  # (n, m, 3)
        V0_ext = np.array([V0]*n)  # (n, m, 3)
        T = O_ext - V0_ext  # (n, m, 3)
        del O_ext, V0_ext

        P = np.cross(D, E2)  # (m, 3)
        E1_ext = np.array([E1]*n)  # (n, m, 3)
        Q = np.cross(T, E1_ext)  # (n, m, 3)
        det = np.einsum('mj,mj->m', P, E1)  # (m,)
        del E1_ext
        
        # 暂时先不写等于零的那个判断，概率太低了
        reverse_det = 1 / det  # (m,)
        reverse_det = np.array([reverse_det]*n)  # (n, m)
        t = reverse_det * np.einsum('nmj,mj->nm', Q, E2)  # (n, m)
        u = reverse_det * np.einsum('mj,nmj->nm', P, T)  # (n, m)
        v = reverse_det * np.einsum('nmj,j->nm', Q, D)  # (n, m)
        is_intersect = np.zeros_like(t, dtype='int16')  # (n, m)
        is_intersect[(t>0) & (u>0) & (v>0) & ((u+v)<1)] = 1  # (n, m)
        intersect_count = np.sum(is_intersect, axis=1)  # (n,)
        return intersect_count
    '''

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

        

if __name__ == '__main__':
    import time
    model = stlmodel('models\\shell_12_large_hole.STL', 1)
    x, y, z = np.meshgrid(
        np.linspace(30, 60, num=40),
        np.linspace(5, 35, num=40), 
        np.linspace(20, 50, num=40)
        )
    grid = np.vstack((x.flatten(), y.flatten(), z.flatten())).T
    model.importGrid(grid)

    t0 = time.time()
    model.calcInModelGridIndex()
    t1 = time.time()
    print('loop method: {:.2f} sec'.format(t1 - t0))
    '''
    print('loop\t| vector')
    for i in range(3):
        t0 = time.time()
        model.calcInModelGridIndex()
        t1 = time.time()
        model.calcInModelGridIndex_vectorized()
        t2 = time.time()
        print('{:.2f}\t| {:.2f}'.format(t1 - t0, t2 - t1))
    '''

    points = grid[np.where(model.in_model_grid_index!=0)]
    from Plot import plotPoints
    plotPoints(points)




    '''
    从STL文件生成点阵模型的算法优化工作
    针对 models\shell_12_large_hole.STL 文件，使用100000点的grid，耗时 62.56 s
    '''
