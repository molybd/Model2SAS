import os
import sys

import numpy as np
from stl import mesh

from utility import convert_coord
import calc_func


class Model:
    '''parent class for single shape model
    '''
    def __init__(self, filename:str) -> None:
        '''init function
        '''
        self.filename = filename
        self.basename = os.path.basename(filename)
        self.modelname = os.path.splitext(self.basename)[0]
        self.modeltype = ''
        self.transform = {}

    def _get_original_boundary(self) -> tuple:
        '''get untransformed boundary coordinates of model
        to be overloaded by child class
        '''
        return np.array((0, 0, 0)), np.array((0, 0, 0))

    def gen_grid_sld(self, grid_x:np.ndarray, grid_y:np.ndarray, grid_z:np.ndarray) -> np.ndarray:
        '''return sld of each grid points
        to be overloaded by child class
        '''
        grid_sld = np.zeros_like(grid_x)
        return grid_sld

    def get_boundary(self) -> tuple:
        '''get boundary coordinates of model
        '''
        boundary_min, boundary_max = self._get_original_boundary()
        xmin, ymin, zmin = tuple(boundary_min)
        xmax, ymax, zmax = tuple(boundary_max)
        x, y, z = np.meshgrid(
            [xmin, xmax],
            [ymin, ymax],
            [zmin, zmax]
            )
        boundary_points = np.vstack((x.flatten(), y.flatten(), z.flatten())).T
        boundary_points = self.apply_transform(boundary_points, on_grid=False)
        boundary_min, boundary_max = np.min(boundary_points, axis=0), np.max(boundary_points, axis=0)
        return boundary_min, boundary_max

    def translate(self, translation_vector:np.ndarray) -> None:
        '''translate model by vector
        Parameters:
            translation: ndarray, vector (x, y, z)
        '''
        self.transform['translate'] = (translation_vector,)

    def rotate(self, axis_center:np.ndarray, axis_direction:np.ndarray, angle:float) -> None:
        '''rotate model around an axis by angle
        Parameters:
            axis_center: ndarray, coordinates where the axis passes
            axis_direction: ndarray, vector describing direction of the axis
            angle: float, rotation angle in radians
        '''
        self.transform['rotate'] = (axis_center, axis_direction, angle)

    def apply_transform(self, points:np.ndarray, on_grid:bool=False) -> np.ndarray:
        '''apply transforms to model, according to self.transform dict
        transform model is realized by reversely transform grid;
        but get boundary need normal transform
        Parameters:
            on_grid: whether the transforms applied on grid or not
        '''
        if on_grid:
            items = list(self.transform.items())
            items.reverse()  # the order must be reversed to apply transform on grid
            for action, args in items:
                if action == 'translate':
                    translation_vector = args[0]
                    points = self._translate(points, -translation_vector)
                elif action == 'rotate':
                    axis_center, axis_direction, angle = args
                    points = self._rotate(points, axis_center, axis_direction, -angle)
        else:
            for action, args in self.transform.items():
                if action == 'translate':
                    points = self._translate(points, *args)
                elif action == 'rotate':
                    points = self._rotate(points, *args)
        return points

    def _translate(self, points:np.ndarray, translation_vector:np.ndarray) -> np.ndarray:
        '''Parameters
            points: ndarray, shape==(n,3), coordinates to translate
            others same as self.translate
        '''
        return points + translation_vector

    def _rotate(self, points:np.ndarray, axis_center:np.ndarray, axis_direction:np.ndarray, angle:float) -> np.ndarray:
        '''Uses the Euler-Rodrigues formula for fast rotations.
        see https://en.wikipedia.org/wiki/Euler%E2%80%93Rodrigues_formula
        Parameters
            points: ndarray, shape==(n,3), coordinates to translate
            others same as self.rotate
        '''
        axis_direction = axis_direction / np.sqrt(np.sum(axis_direction**2)) # make axis unit vector
        kx, ky, kz = tuple(axis_direction)
        a, b, c, d = np.cos(angle/2), kx*np.sin(angle/2), ky*np.sin(angle/2), kz*np.sin(angle/2)
        rotation_matrix = np.array([
            [a**2 + b**2 - c**2 - d**2, 2*(b*c - a*d), 2*(b*d + a*c)],
            [2*(b*c + a*d), a**2 + c**2 - b**2 - d**2, 2*(c*d - a*b)],
            [2*(b*d - a*c), 2*(c*d + a*b), a**2 + d**2 - b**2 - c**2]
        ])
        # make points the origin and than rotate, then translate back
        points = self._translate(points, -axis_center)
        points = np.einsum('ij,nj->ni', rotation_matrix, points)
        points = self._translate(points, axis_center)
        return points


class StlModel(Model):
    '''class for model from stl files
    '''
    def __init__(self, filename: str, sld:float=1, centering:bool=True) -> None:
        '''default uniform sld value
        '''
        super().__init__(filename)
        self.modeltype = 'stl'
        self.sld=sld
        self.mesh = mesh.Mesh.from_file(self.filename)
        if centering:  # move model center to (0,0,0)
            center = self.mesh.get_mass_properties()[1]
            self.mesh.translate(-center)

    def _get_original_boundary(self) -> tuple:
        '''get untransformed boundary coordinates of stlmodel
        '''
        vectors = self.mesh.vectors
        x, y, z = vectors[:,:,0].flatten(), vectors[:,:,1].flatten(), vectors[:,:,2].flatten()
        boundary_points = np.vstack((x, y, z)).T
        boundary_min, boundary_max = np.min(boundary_points, axis=0), np.max(boundary_points, axis=0)
        return boundary_min, boundary_max

    def gen_grid_sld(self, grid_x:np.ndarray, grid_y:np.ndarray, grid_z:np.ndarray) -> np.ndarray:
        '''return sld of each grid points
        '''
        x, y, z = grid_x.flatten(), grid_y.flatten(), grid_z.flatten()
        grid = np.vstack((x, y, z)).T
        grid = self.apply_transform(grid, on_grid=True)

        # determine whether points inside the model
        ray = np.random.random((3,)).astype(np.float32)    # use random ray. use ray like [1,1,1] may cause some misjudge
        if np.sum(ray) <= np.finfo(np.float32).eps:  # in case that all coordinates are 0 so add 0.01, which is almost impossible
            ray = np.array([0.23782647, 0.90581098, 0.34623647], dtype=np.float32)
        
        vectors = self.mesh.vectors.astype(np.float32)
        intersect_count = calc_func.moller_trumbore_intersect_count(grid, ray, vectors)
        index = intersect_count % 2   # 1 is in, 0 is out
        sld_index = self.sld * index

        grid_sld = sld_index.reshape(grid_x.shape)
        return grid_sld

    def set_sld(self, sld:float) -> None:
        '''set sld value of stlmodel
        '''
        self.sld = sld

class MathModel(Model):
    '''class for model from math description .py file
    '''
    def __init__(self, filename: str) -> None:
        '''init, import mathmodel module
        '''
        super().__init__(filename)
        self.modeltype = 'math'
        abspath = os.path.abspath(filename)
        dirname = os.path.dirname(abspath)
        sys.path.append(dirname)
        mathmodel_module = __import__(self.modelname)
        self.mathmodel_obj = mathmodel_module.SpecificMathModel()

    def _get_original_boundary(self) -> tuple:
        '''get untransformed boundary coordinates of mathmodel
        '''
        return self.mathmodel_obj.get_boundary()

    def gen_grid_sld(self, grid_x:np.ndarray, grid_y:np.ndarray, grid_z:np.ndarray) -> np.ndarray:
        '''return sld of each grid points
        '''
        x, y, z = grid_x.flatten(), grid_y.flatten(), grid_z.flatten()
        grid = np.vstack((x, y, z)).T
        grid = self.apply_transform(grid, on_grid=True)
        x, y, z = tuple(grid.T)

        coord = self.mathmodel_obj.coord
        u, v, w = convert_coord(x, y, z, 'car', coord)
        grid_points_in_coord = np.vstack((u, v, w)).T
        sld_index = self.mathmodel_obj.sld(grid_points_in_coord)
        grid_sld = sld_index.reshape(grid_x.shape)
        return grid_sld

    def change_parameter(self, param_name:str, value:float) -> None:
        '''change the parameter of math model in mathmodel_obj.params dict
        '''
        self.mathmodel_obj.params[param_name] = value




if __name__ == '__main__':
    pass

