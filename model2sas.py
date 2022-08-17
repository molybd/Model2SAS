import os

import numpy as np
from model import MathModel, StlModel
import calc_func


class Model2Sas:
    '''core class of Model2SAS
    generate lattice model and calculate SAS curve
    '''

    def __init__(self, proj_name:str) -> None:
        '''set up a new project with name proj_name
        '''
        self.name = proj_name
        self.models = {}

    def import_model(self, filename:str) -> None:
        '''import model file, only accept .stl and .py file
        '''
        extname = os.path.splitext(filename)[-1].lower()
        if extname == '.stl':
            model = StlModel(filename)
        elif extname == '.py':
            model = MathModel(filename)
        modelkey = self._gen_modelkey(model.modelname)
        self.models[modelkey] = model

    def del_model(self, modelkey:'str') -> None:
        '''delete a model in self.models by its key
        '''
        del self.models[modelkey]

    def _gen_modelkey(self, modelname:str) -> str:
        '''generate key in self.models dict
        '''
        i = 0
        modelkey = '{}_{}'.format(modelname, str(i))
        existing_modelkeys = list(self.models.keys())
        while modelkey in existing_modelkeys:
            i += 1
            modelkey = '{}_{}'.format(modelname, str(i))
        return modelkey

    def gen_suggested_parameters(self) -> tuple:
        '''To evaluate optimal parameter values based on my experience.
        e.g. smaller interval surely gives better results but maybe unnecessary
        and may cause error since it uses much larger RAM and computing time. 
        '''
        # i means interval here
        boundary_min, boundary_max = self.gen_combined_boundary()
        length = np.max(boundary_max-boundary_min)
        qmin, qmax = (1/length)*0.2, (1/length)*25
        smin, smax = qmin/(2*np.pi), qmax/(2*np.pi)
        i_expected_max = 1/(2*smax)
        i_times_ns_expected_min = 1/smin
        ns = 600  # grid number after fft, same as n_s in fft method
        i_expected_min = i_times_ns_expected_min / ns
        i = min(i_expected_min, i_expected_max)
        sugg_param_dict = {
            'q_min': qmin,
            'q_max': qmax,
            'interval': i
        }
        return sugg_param_dict

    def gen_grid(self, interval:float, boundary_min:np.ndarray, boundary_max:np.ndarray) -> tuple:
        '''generate mesh grid according to boundary,
        cubic grid with same interval, for FFT use
        '''
        center = (boundary_min + boundary_max)/2
        length = np.max(boundary_max-boundary_min)
        n = int(length/interval+1)
        l = np.linspace(-0.5, 0.5, num=n) * length
        x, y, z = l+center[0], l+center[1], l+center[2]
        grid_x, grid_y, grid_z = np.meshgrid(x, y, z)
        return grid_x, grid_y, grid_z

    def gen_combined_boundary(self) -> tuple:
        '''generate overall boundary of combined models of all in self.models
        '''
        boundary_points = []
        for model in self.models.values():
            boundary_points.append(model.get_boundary())
        boundary_points = np.array(boundary_points) # shape==(n, 2, 3)
        boundary_min = np.min(boundary_points[:,0,:], axis=0)
        boundary_max = np.max(boundary_points[:,1,:], axis=0)
        return boundary_min, boundary_max

    def gen_combined_model(self, interval:float) -> tuple:
        '''generate lattice model of combined models of all in self.models
        '''
        boundary_min, boundary_max = self.gen_combined_boundary()
        grid_x, grid_y, grid_z = self.gen_grid(interval, boundary_min, boundary_max)
        grid_sld = np.zeros_like(grid_x)
        for model in self.models.values():
            grid_sld += model.gen_grid_sld(grid_x, grid_y, grid_z)
        return grid_x, grid_y, grid_z, grid_sld

    def gen_sas(self, grid_x:np.ndarray, grid_y:np.ndarray, grid_z:np.ndarray, grid_sld:np.ndarray, q:np.ndarray, method='fft', **kwargs) -> tuple:
        '''generate SAS curve
        Parameters
            grid_x: ndarray, shape==(n, n, n)
            grid_y: ndarray, shape==(n, n, n)
            grid_z: ndarray, shape==(n, n, n)
            grid_sld: ndarray, shape==(n, n, n)
            q: 1darray, q values to calculate
            method: 'fft', 'sph harm' or 'debye func'
        Return
            q: 1darray, may not be same as input q when use fft method
            I: theoretical intensity values, same shape as q
        '''
        if 'sph' in method:
            x, y, z, sld = grid_x.flatten(), grid_y.flatten(), grid_z.flatten(), grid_sld.flatten()
            x, y, z, sld = x[np.where(sld!=0)], y[np.where(sld!=0)], z[np.where(sld!=0)], sld[np.where(sld!=0)]
            q, I = calc_func.sas_sphharm(x, y, z, sld, q, **kwargs)
        elif 'debye' in method:
            x, y, z, sld = grid_x.flatten(), grid_y.flatten(), grid_z.flatten(), grid_sld.flatten()
            x, y, z, sld = x[np.where(sld!=0)], y[np.where(sld!=0)], z[np.where(sld!=0)], sld[np.where(sld!=0)]
            q, I = calc_func.sas_debyefunc(x, y, z, sld, q)
        else: # use fft method for default
            interval = grid_x[0,1,0] - grid_x[0,0,0]
            q, I = calc_func.sas_fft(grid_sld, interval, q, **kwargs)
        return q, I


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    
    # init cuda and numba for faster later
    calc_func.USE_TORCH = True
    calc_func.DEVICE = 'cuda'
    init = Model2Sas('init')
    init.import_model('models/torus.stl')
    grid_x, grid_y, grid_z, grid_sld = init.gen_combined_model(2)
    q_init = np.logspace(-2, 0, num=200)
    q_init, I_init = init.gen_sas(grid_x, grid_y, grid_z, grid_sld, q_init, method='fft')
    print('='*40)


    proj = Model2Sas('test')
    proj.import_model('models/torus.stl')
    proj.import_model('models/shell_12_large_hole.STL')
    proj.import_model('mathmodel_template.py')
    proj.models['torus_0'].translate(np.array([0, 0, 10]))
    proj.models['torus_0'].rotate(np.array([0, 0, 10]), np.array([1, 1, 0]), np.pi/4)
    grid_x, grid_y, grid_z, grid_sld = proj.gen_combined_model(1)
    print('='*40)

    #figure = plt.figure()
    #axes = mplot3d.Axes3D(figure)
    #x, y, z = grid_x[np.where(grid_sld != 0)], grid_y[np.where(grid_sld != 0)], grid_z[np.where(grid_sld != 0)]
    #axes.scatter(x, y, z)
    #plt.show()
    
    q = np.logspace(-2, 0, num=200)

    calc_func.USE_TORCH = False
    q1, I1 = proj.gen_sas(grid_x, grid_y, grid_z, grid_sld, q, method='fft')
    print('='*40)
    calc_func.USE_TORCH = True
    calc_func.DEVICE = 'cuda'
    q1, I1 = proj.gen_sas(grid_x, grid_y, grid_z, grid_sld, q, method='fft')
    print('='*40)
    q2, I2 = proj.gen_sas(grid_x, grid_y, grid_z, grid_sld, q, method='sph harm')
    print('='*40)
    q3, I3 = proj.gen_sas(grid_x, grid_y, grid_z, grid_sld, q, method='debye_func')
    print('='*40)
    plt.plot(q1, I1)
    plt.plot(q2, I2)
    plt.plot(q3, I3)
    plt.yscale('log')
    plt.xscale('log')
    plt.savefig('test.png')
    
