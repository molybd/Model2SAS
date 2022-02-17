import numpy as np

class specific_mathmodel:

    def __init__(self):
        self.params = {"r1": 20, "r2": 30}
        self.coord = "sph"  # 'xyz' or 'sph' or 'cyl'
        self.boundary_min, self.boundary_max = self.genBoundary()
        # must have these 4 attributes

    def genBoundary(self):
        r1 = self.params["r1"]
        r2 = self.params["r2"]  # params statement

        boundary_min, boundary_max = -1*np.array([1, 1, 1]), np.array([1, 1, 1])
        boundary_max = r2 * boundary_max
        boundary_min = -1 * boundary_max  # boundary statement
        self.boundary_min, self.boundary_max = boundary_min, boundary_max
        return boundary_min, boundary_max

    def shape(self, grid_in_coord):
        self.grid_in_coord = grid_in_coord
        self.u, self.v, self.w = grid_in_coord[:,0], grid_in_coord[:,1], grid_in_coord[:,2]
        r, theta, phi = self.u, self.v, self.w  # u, v, w to certain coordinate variables
        
        # set parameters
        r1 = self.params["r1"]
        r2 = self.params["r2"]  # params statement

        in_model_grid_index = np.zeros_like(self.u)
        in_model_grid_index[(r>=r1) & (r<=r2)] = 1

        self.in_model_grid_index = in_model_grid_index
        return in_model_grid_index

    def sld(self):
        r, theta, phi = self.u, self.v, self.w  # u, v, w to certain coordinate variables

        # set parameters
        r1 = self.params["r1"]
        r2 = self.params["r2"] # params statement

        sld = np.ones_like(self.u)
        sld = 6.34 * sld
        
        sld_grid_index = sld * self.in_model_grid_index
        return sld_grid_index
        