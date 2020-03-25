# -*- coding: UTF-8 -*-

import numpy as np
from stl import mesh
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from scipy.special import sph_harm, spherical_jn, jv
import os, sys
from multiprocessing import Pool, cpu_count
import inspect

# attention: in this program, spherical coordinates are (r, theta, phi) |theta: 0~pi ; phi: 0~2pi
#
# unsolved problem:
# Q1. SAS curve is wrong when there are points coordinates less than zero
#     have add some solution, but don't know the reason behind
# A1. the problem should result from the func xyz2sph, where the range of arctan is not 2pi, is [-pi/2, pi/2]
#     solution: use np.arctan2() method instead of np.arctan()

class model2sas:
    'class to read 3D model from file and generate PDB file and SAS curve'

    def __init__(self, filename, interval=None, procNum=None, modelName=None, autoGenPoints=True, *args, **kwargs):
        self.file_abspath = os.path.abspath(filename)                               # first convert to abs path
        self.inputFileDir = os.path.dirname(self.file_abspath)                      # only file name, without dir info
        self.inputFileName = os.path.basename(self.file_abspath)                    # only absolute dir of the file
        self.inputFileType = self.inputFileName.split('.')[-1].lower()              # file type extension, e.g. stl, py etc.
        self.modelname = self.__determineModelName(modelName, self.inputFileName)   # determine model name, without extention
        self.stlModelMesh = None                                                    # initial mesh of stl file
        self.meshgrid = np.array([])                                                # the initial meshgrid
        self.pointsInModel = np.array([])                                           # points coordinates inside the model
        self.sasCurve = np.array([])                                                # SAS curve calculated
        self.workingDir = os.getcwd()                                               # cwd
        
        # process number in paralell computing, default is using 60% CPU maximum
        if procNum == None:
            self.procNum = min(round(0.6*cpu_count()), cpu_count()-1)
        else:
            self.procNum = procNum
        
        if autoGenPoints:
            if self.inputFileType != 'py':
                self.buildFromFile(interval=interval)
            else:
                self.buildFromMath(interval=interval)

    def __determineModelName(self, modelname, filename):
        if modelname == None:
            modelname = '.'.join(filename.split('.')[:-1])
        else:
            modelname = modelname
        return modelname

    def generateMeshgrid(self, xmin, xmax, ymin, ymax, zmin, zmax):
        xscale = np.linspace(xmin, xmax, num=int((xmax-xmin)/self.interval+1))
        yscale = np.linspace(ymin, ymax, num=int((ymax-ymin)/self.interval+1))
        zscale = np.linspace(zmin, zmax, num=int((zmax-zmin)/self.interval+1))
        x, y, z = np.meshgrid(xscale, yscale, zscale)
        x, y, z = x.reshape(x.size,1), y.reshape(y.size,1), z.reshape(z.size,1)
        self.meshgrid = np.hstack((x, y, z))
        return self.meshgrid

    # new method, about 1000 times faster than old one...
    # make full use of numpy
    # calculate all the points intersect with 1 reiangle using multi-D array
    def isIntersect(self, origins, ray, triangle):
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
        

    # build points model from file
    # supported file type: .stl, .xyz, .txt(points array from np.savetxt() )
    def buildFromFile(self, interval=None, modelname=None):
        os.chdir(self.inputFileDir)
        filename = self.inputFileName
        filetype = self.inputFileType # file extension in lower cases
        
        if filetype == 'stl':
            self.stlModelMesh = mesh.Mesh.from_file(filename)
            vectors = self.stlModelMesh.vectors
            xmin, xmax, ymin, ymax, zmin, zmax = np.min(vectors[:,:,0]), np.max(vectors[:,:,0]), np.min(vectors[:,:,1]), np.max(vectors[:,:,1]), np.min(vectors[:,:,2]), np.max(vectors[:,:,2])
            if interval == None:
                self.interval = min([xmax-xmin, ymax-ymin, zmax-zmin]) / 20
            else:
                self.interval = interval
            self.generateMeshgrid(xmin, xmax, ymin, ymax, zmin, zmax)

            # (solved) this must be the slowest process in the whole program ! 
            # (solved) must be a way to accelerate, still working on it...
            # (solved) no need to use multiprocessing
            # 
            # accelerated * 1
            # about 1000 times faster
            ray = np.random.rand(3) + 0.01     # in case that all coordinates are 0, which is almost impossible
            intersect_count = np.zeros(self.meshgrid.shape[0])  # same number as the points in initial meshgrid
            # For each triangle in model mesh, determine that every points
            # in initial meshgrid intersect with this triangle or not.
            for triangle in vectors:
                intersect_count += self.isIntersect(self.meshgrid, ray, triangle)
            indexInOrOut = intersect_count % 2   # index to judge a point is in or out of the model, 1 is in, 0 is out

            pointsInModelList = []
            for i in range(len(indexInOrOut)):
                if indexInOrOut[i] > 0:
                    pointsInModelList.append(self.meshgrid[i])

            self.pointsInModel = np.array(pointsInModelList)  # shape = (points number, 3)

        elif filetype == 'xyz':
            lst = []
            with open(filename, 'r') as f:
                for line in f.readlines():
                    if line[0] != '#':
                        point = [float(i) for i in line.split()[1:4]]
                        lst.append(point)
            self.pointsInModel = np.array(lst)
            self.interval = interval
            
        elif filetype == 'txt':
            self.pointsInModel = np.loadtxt(filename)
            self.interval = interval
        
        os.chdir(self.workingDir)
        return self.pointsInModel


    # to generate a 3D model from a mathematical description
    # math description is in seperated file module.py
    # function in module.py return True or False if a point is in the model
    # boundaryList is [xmin, xmax, ymin, ymax, zmin, zmax]
    # coord is 'xyz' 
    # or 'sph'(r, theta, phi)|theta: 0~pi ; phi: 0~2pi
    # or 'cyl'(rho, phi, z)
    def buildFromMath(self, interval=None, useDefault=True, **kwargs):
        os.chdir(self.inputFileDir)

        sys.path.append(self.inputFileDir)   # add dir to sys.path, then we can directly import the py file
        modelModule = '.'.join(self.inputFileName.split('.')[:-1])
        mathModel = __import__(modelModule)

        # read function arguments
        # inspect.getargspec() is deprecated since Python 3.0
        # args = inspect.getargspec(mathModel.model) 
        sig = inspect.signature(mathModel.model)
        params = sig.parameters

        # read coordinates from math model file
        coord = params['coord'].default

        # means use default value in model file
        if useDefault:
            
            # read boundaryList from math model file
            boundaryList = params['boundary_xyz'].default
            [xmin, xmax, ymin, ymax, zmin, zmax] = boundaryList

            # set interval
            if interval == None:
                self.interval = min(np.abs([xmax-xmin, ymax-ymin, zmax-zmin])) / 20
            else:
                self.interval = interval
            self.generateMeshgrid(xmin, xmax, ymin, ymax, zmin, zmax)

            # convert coordinates
            if coord == 'xyz':
                points = self.meshgrid
            elif coord == 'sph':
                points = self.xyz2sph(self.meshgrid)
            elif coord == 'cyl':
                points = self.xyz2cyl(self.meshgrid)

            pointsInModelList = []
            inModelIndex = mathModel.model(points)
            for i in range(len(inModelIndex)):
                if inModelIndex[i] > 0:
                    pointsInModelList.append(self.meshgrid[i,:])
            self.pointsInModel = np.array(pointsInModelList)

        else:
            
            if 'boundary_xyz' in kwargs.keys():
                # read boundaryList from custom input
                boundaryList = kwargs['boundary_xyz']
            else:
                # read boundaryList from math model file
                boundaryList = params['boundary_xyz'].default
            [xmin, xmax, ymin, ymax, zmin, zmax] = boundaryList

            # set interval
            if interval == None:
                self.interval = min([xmax-xmin, ymax-ymin, zmax-zmin]) / 20
            else:
                self.interval = interval
            self.generateMeshgrid(xmin, xmax, ymin, ymax, zmin, zmax)

            # convert coordinates
            if coord == 'xyz':
                points = self.meshgrid
            elif coord == 'sph':
                points = self.xyz2sph(self.meshgrid)
            elif coord == 'cyl':
                points = self.xyz2cyl(self.meshgrid)

            # generate custom args string
            arg_string_list = []
            for item in kwargs.items():
                arg_string_list.append('{}={}'.format(item[0], item[1]))
            arg_string = ','.join(arg_string_list)

            pointsInModelList = []

            # 这里是 exec() 使用的的一个坑，要弄清楚变量命名空间以及exec()作用方式等问题
            g, l = globals().copy(), locals().copy()
            exec(
                'inModelIndex = mathModel.model(points, {})'.format(arg_string),
                g,
                l
            )
            inModelIndex = l['inModelIndex']
            for i in range(len(inModelIndex)):
                if inModelIndex[i] > 0:
                    pointsInModelList.append(self.meshgrid[i,:])
            self.pointsInModel = np.array(pointsInModelList)


        os.chdir(self.workingDir)
        return self.pointsInModel


    # save points in model in a file
    # you must provide at least file type or filename
    # otherwise by default it will save a pdb file
    def savePointsInModel(self, filetype='pdb', filename=None):
        os.chdir(self.inputFileDir)

        if filename == None:
            filename = '{}_interval={}.{}'.format(
                self.modelname,
                str(int(round(self.interval))),
                filetype
            )
            # if no filename is assigned, output file will be saved in the same dir as input file
            filename = os.path.join(self.inputFileDir, filename)
        else:
            filetype = filename.split('.')[-1]

        if filetype == 'txt':
            np.savetxt(filename, self.pointsInModel)
        elif filetype == 'pdb':
            self.savePDBFile(filename=filename)
        elif filetype == 'xyz':
            self.saveXYZFile(filename=filename)
        
        os.chdir(self.workingDir)

    def saveXYZFile(self, filename='', head='created by program Model2SAS', atom='CA'):
        if filename == '':
            filename = self.modelname + '.xyz'
        with open(filename, 'w') as f:
            s = '#' + head + '\n'
            for point in self.pointsInModel:
                s += '{}\t{}\t{}\t{}\n'.format(atom, point[0], point[1], point[2])
            f.write(s)

    def savePDBFile(self, filename=None, atom='CA', occupancy=1.0, tempFactor=20.0):
        if filename == None:
            filename = self.modelname + '.pdb'
        self.PDBfilename = os.path.basename(filename)
        with open(filename, 'w') as f:
            s = 'REMARK 265 EXPERIMENT TYPE: THEORETICAL MODELLING\n'
            for i in range(len(self.pointsInModel)):
                x = '{:.2f}'.format(self.pointsInModel[i, 0])
                y = '{:.2f}'.format(self.pointsInModel[i, 1])
                z = '{:.2f}'.format(self.pointsInModel[i, 2])
                s += 'ATOM  {:5d} {:<4} ASP A{:4d}    {:>8}{:>8}{:>8}{:>6}{:>6} 0 2 201\n'.format(int(i), atom, i%10, x, y, z, str(occupancy), str(tempFactor))
            f.write(s)

    def plotSTLMeshModel(self, plot=True):
        # Create a new plot
        fig = plt.figure()
        axes = mplot3d.Axes3D(fig)

        # Load the STL files and add the vectors to the plot
        axes.add_collection3d(mplot3d.art3d.Poly3DCollection(self.stlModelMesh.vectors))
    
        # Auto scale to the mesh size
        scale = self.stlModelMesh.points.flatten()
        axes.auto_scale_xyz(scale, scale, scale)

        # Show the plot to the screen
        if plot:
            plt.show()
        return fig

    def plotPointsInModel(self, plot=True):
        # Create a new plot
        fig = plt.figure()
        axes = mplot3d.Axes3D(fig)

        axes.scatter(self.pointsInModel[:,0], self.pointsInModel[:,1], self.pointsInModel[:,2], color='k')
        # Show the plot to the screen
        if plot:
            plt.show()
        return fig

    def genSasCurve_Crysol(self, qmax=1, qNum=256, lmax=50, crysolPath=None):
        self.lmax = lmax
        self.savePointsInModel(filetype='pdb')
        os.chdir(self.inputFileDir)
        # first delete all files genetated by crysol before
        checkFilename = self.PDBfilename[:-4]
        filelist = list(os.listdir(self.inputFileDir))
        for i in range(len(filelist)):
            if checkFilename in filelist[i] and filelist[i].split('.')[-1] in ['abs', 'alm', 'int', 'log']:
                os.remove(
                    os.path.join(self.inputFileDir, filelist[i])
                )

        if crysolPath == None:
            os.system('crysol {} -lm {} -fb 18 -sm {} -ns {} -un 1'.format(self.PDBfilename, lmax, qmax, qNum))
        else:
            os.system('\"{}\" {} -lm {} -fb 18 -sm {} -ns {} -un 1'.format(crysolPath, self.PDBfilename, lmax, qmax, qNum))
        intfile = self.PDBfilename[:-4] + '00.int'
        crysolOutput = np.loadtxt(intfile, skiprows=1)
        self.sasCurve = crysolOutput[:, :2]
        os.chdir(self.workingDir)
        return self.sasCurve

    # transfer points coordinates from cartesian coordinate to cylindrical coordinate
    # points_xyz must be array([[x0,y0,z0], [x1,y1,z1], [x1,y1,z1], ...])
    # returned points_sph is array([[r0, phi0, z0], [r1, phi1, z1], [r2, phi2, z2], ...])
    def xyz2cyl(self, points_xyz):
        r = np.linalg.norm(points_xyz[:,:2], axis=1)
        phi = np.arctan2(points_xyz[:,1], points_xyz[:,0])
        z = points_xyz[:,2]
        points_cyl = np.vstack((r, phi, z)).T
        return points_cyl

    # transfer points coordinates from cartesian coordinate to spherical coordinate
    # points_xyz must be array([[x0,y0,z0], [x1,y1,z1], [x1,y1,z1], ...])
    # returned points_sph is array([[r0, theta0, phi0], [r1, theta1, phi1], [r1, theta1, phi1], ...])
    def xyz2sph(self, points_xyz):
        epsilon=1e-100
        r = np.linalg.norm(points_xyz, axis=1)
        theta = np.arccos(points_xyz[:,2] / (r+epsilon))
        phi = np.arctan2(points_xyz[:,1], points_xyz[:,0]) # range [-pi, pi]
        phi = phi + np.sign((np.sign(-1*phi) + 1))*2*np.pi # convert range to [0, 2pi]
        points_sph = np.vstack((r, theta, phi)).T
        return points_sph  # theta: 0~pi ; phi: 0~2pi

    def sph2xyz(self, points_sph):
        r, theta, phi = points_sph[:,0], points_sph[:,1], points_sph[:,2]
        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)
        points_xyz = np.vstack((x, y, z)).T
        return points_xyz


    # unit sphere form factor actually results in wrong outcomes
    # so I delete it ...
    #
    # new method using matrix calculation to accelerate
    # it actually only becomes a little bit faster (about 10%)... 
    #
    def Alm(self, args):
        q, points_sph, l, m = args
        A = 0
        # p_sph: array[r, theta, phi]; theta: 0~pi, phi: 0~2pi
        r, theta, phi = points_sph[:, 0], points_sph[:, 1], points_sph[:, 2] # theta: 0~pi ; phi: 0~2pi
        q = q.reshape(q.shape[0], 1)
        r = r.reshape(1, r.shape[0])
        A = spherical_jn(l, q*r) * sph_harm(m, l, phi, theta)
        A = A.sum(axis=1)
        return 4 * np.pi * complex(0,1)**l * A

    # used in func genSasCurve()
    # points in spherical coordinates
    def Iq(self, q, points_sph, lmax):
        def gen_args(lmax):
            for l in range(lmax+1):
                for m in range(-l, l+1):
                    yield (q, points_sph, l, m)
        # use mult-processing to accelarate
        # use map_async() instead of apply_async()
        pool = Pool(self.procNum)
        result = pool.map_async(self.Alm, gen_args(lmax))
        pool.close()
        pool.join()
        I = abs(np.array(result.get()))**2
        I = I.sum(axis=0)
        return I

    def genSasCurve(self, qmin=0.01, qmax=1, qnum=200, lmax=50):
        self.lmax = lmax
        points_sph = self.xyz2sph(self.pointsInModel)
        q = np.linspace(qmin, qmax, num=qnum)
        I = self.Iq(q, points_sph, lmax)
        self.sasCurve = np.vstack((q, I)).T
        return self.sasCurve

    def saveSasCurve(self):
        os.chdir(self.inputFileDir)
        filename = self.modelname + '_saxs.dat'
        header = 'theoretical SAXS curve of {} model\ninterval between points = {}\nl_max in spherical harmonics = {}\nmodel generated by program Model2SAS\n'.format(self.modelname, self.interval, self.lmax)
        header += '\nq\tI'
        np.savetxt(filename, self.sasCurve, header=header)
        os.chdir(self.workingDir)

    def plotSasCurve(self, show=True, save=False, figsize=None, dpi=None, figname=None):
        q = self.sasCurve[:, 0]
        I = self.sasCurve[:, 1]

        if save:
            fig = plt.figure(figsize=figsize, dpi=dpi, facecolor='white')
        else:
            fig = plt.figure()
        ax1 = fig.add_subplot(111)

        ax1.plot(q, I, '-', label=self.modelname)

        ax1.set_xscale('log')
        ax1.set_yscale('log')

        ax1.set_xlabel(r'Q $(\AA^{-1})$', fontsize=13)
        ax1.set_ylabel(r'Intensity (a.u.)', fontsize=13)

        ax1.legend(fontsize=11, frameon=False)

        if show:
            plt.show()
        if save:
            cwd = os.getcwd()
            os.chdir(self.inputFileDir)
            fig.savefig(figname)
            os.chdir(cwd)

        return fig

if __name__ == '__main__':

    modelType = 'math'

    if modelType == 'stlfile':
        model = model2sas('mdoels/sphere_phi100.STL') # this will generate points model automatically
        #model.plotPointsInModel()
        model.savePointsInModel(filetype='pdb')
        model.genSasCurve_Crysol()
        model.plotSasCurve()
    elif modelType == 'math':
        model = model2sas('models\\ellipsoid_model.py', autoGenPoints=False)
        model.buildFromMath(interval=2, useDefault=False, a=20, b=20, c=20, boundary_xyz=[-20, 20, -20, 20, -20, 20])
        model.plotPointsInModel()
        model.genSasCurve_Crysol()
        model.plotSasCurve()
        