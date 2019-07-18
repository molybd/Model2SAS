# -*- coding: UTF-8 -*-

import numpy as np
from stl import mesh
from mpl_toolkits import mplot3d
from matplotlib import pyplot
from scipy.special import sph_harm, spherical_jn, jv
import os
from multiprocessing import Pool
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

    def __init__(self, procNum=1, *args, **kwargs):
        self.modelname = ''
        self.meshgrid = np.array([])
        self.pointsInModel = np.array([])
        self.stlModelMesh = None
        self.sasCurve = np.array([])
        self.procNum = procNum

    def __determineModelName(self, modelname, filename):
        if modelname == None:
            if '/' in filename:
                self.modelname = filename.split('/')[-1].split('.')[-2]
            elif '\\' in filename:
                self.modelname = filename.split('\\')[-1].split('.')[-2]
            else:
                self.modelname = filename.split('.')[-2]
        else:
            self.modelname = modelname

    def generateMeshgrid(self, xmin, xmax, ymin, ymax, zmin, zmax):
        xscale = np.linspace(xmin, xmax, num=int((xmax-xmin)/self.interval+1))
        yscale = np.linspace(ymin, ymax, num=int((ymax-ymin)/self.interval+1))
        zscale = np.linspace(zmin, zmax, num=int((zmax-zmin)/self.interval+1))
        x, y, z = np.meshgrid(xscale, yscale, zscale)
        x, y, z = x.reshape(x.size,1), y.reshape(y.size,1), z.reshape(z.size,1)
        self.meshgrid = np.hstack((x, y, z))
        return self.meshgrid

    def isIntersect(self, origin, ray, triangle):
        intersec = False
        O = origin
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
        if det != 0:
            intersectPoint = (1/det) * np.hstack((np.dot(Q, E2), np.dot(P, T), np.dot(Q, D)))
        else:
            intersec = False
            intersectPoint = np.array([np.nan, np.nan, np.nan])
        t, u, v = intersectPoint[0], intersectPoint[1], intersectPoint[2]
        if t >= 0 and u >= 0 and v >= 0 and (u+v) <= 1:
            intersec = True
        else:
            intersec = False
        return intersec, intersectPoint

    def isPointInSTLModel(self, point, stlModelMesh, ray=np.array([1,1,1]), eps=0.001):
        intersectPointList = []
        for triangle in stlModelMesh.vectors:
            intersect, intersectPoint = self.isIntersect(point, ray, triangle)
            if intersect:
                intersectPointList.append(intersectPoint)
        noOverlapPointList = []
        if len(intersectPointList) > 1:
            for i in range(len(intersectPointList)):
                overlap = False
                for j in range(i+1, len(intersectPointList)):
                    if np.sqrt(np.sum(np.square(intersectPointList[i]-intersectPointList[j]))) <= eps:
                        overlap = True
                        break
                if not overlap:
                    noOverlapPointList.append(intersectPointList[i])
        else:
            noOverlapPointList = intersectPointList
        if len(noOverlapPointList)%2 == 0:
            isInModel = False
        else:
            isInModel = True
        return isInModel, np.array(noOverlapPointList)

    # for the usage of multiprocessing only
    def ptsInSTLModel(self, pts):
        ptsInModelList = []
        for pt in pts:
            if self.isPointInSTLModel(pt, self.stlModelMesh)[0]:
                ptsInModelList.append(pt)
        return ptsInModelList

    # build points model from file
    # supported file type: .stl, .xyz, .txt(points array from np.savetxt() )
    def buildFromFile(self, filename, interval=None, modelname=None):
        self.__determineModelName(modelname, filename)

        filetype = filename.split('.')[-1].lower() # file extension in lower cases
        
        if filetype == 'stl':
            self.stlModelMesh = mesh.Mesh.from_file(filename)
            vectors = self.stlModelMesh.vectors
            xmin, xmax, ymin, ymax, zmin, zmax = np.min(vectors[:,:,0]), np.max(vectors[:,:,0]), np.min(vectors[:,:,1]), np.max(vectors[:,:,1]), np.min(vectors[:,:,2]), np.max(vectors[:,:,2])
            if interval == None:
                self.interval = min([xmax-xmin, ymax-ymin, zmax-zmin]) / 20
            else:
                self.interval = interval
            self.generateMeshgrid(xmin, xmax, ymin, ymax, zmin, zmax)

            # this must be the slowest process in the whole program !
            # must be a way to accelerate, still working on it...
            # use multiprocessing to accelerate
            pointsInModelList = []
            multip_result_list = []
            length = len(self.meshgrid)//self.procNum + 1
            pool = Pool(self.procNum)
            for i in range(self.procNum):
                pts = self.meshgrid[i*length: (i+1)*length]
                multip_result_list.append(pool.apply_async(self.ptsInSTLModel, args=(pts,)))
            pool.close()
            pool.join()
            for item in multip_result_list:
                pointsInModelList += item.get()

            self.pointsInModel = np.array(pointsInModelList)

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
        
        return self.pointsInModel


    # to generate a 3D model from a mathematical description
    # math description is in seperated file module.py
    # function in module.py return True or False if a point is in the model
    # boundaryList is [xmin, xmax, ymin, ymax, zmin, zmax]
    # coord is 'xyz' or 'sph'(r, theta, phi)|theta: 0~pi ; phi: 0~2pi
    def buildFromMath(self, modelname, module, function, boundaryList, coord='xyz', interval=None):
        self.modelname = modelname
        xmin, xmax, ymin, ymax, zmin, zmax = boundaryList[0], boundaryList[1], boundaryList[2], boundaryList[3], boundaryList[4], boundaryList[5]
        if interval == None:
            self.interval = min([xmax-xmin, ymax-ymin, zmax-zmin]) / 20
        else:
            self.interval = interval
        self.generateMeshgrid(xmin, xmax, ymin, ymax, zmin, zmax)
        pointsInModelList = []

        # read coordinates from math function
        MathDescription = __import__(module)
        args = inspect.getargspec(eval('MathDescription.{}'.format(function)))
        coord = args.defaults[-1]

        if coord == 'xyz':
            points = self.meshgrid
        elif coord == 'sph':
            points = self.xyz2sph(self.meshgrid)
        elif coord == 'cyl':
            points = self.xyz2cyl(self.meshgrid)

        for i in range(len(self.meshgrid)):
            p = points[i,:]
            conditional_statement = 'MathDescription.{}(p)'.format(function)
            if eval(conditional_statement):
                pointsInModelList.append(self.meshgrid[i,:])
        self.pointsInModel = np.array(pointsInModelList)
        return self.pointsInModel


    def savePointsInModel(self):
        filename = '{}_interval={}.txt'.format(self.modelname, self.interval)
        np.savetxt(filename, self.pointsInModel)

    def saveXYZFile(self, filename='', head='created by limu', atom='CA'):
        if filename == '':
            filename = self.modelname + '.xyz'
        with open(filename, 'w') as f:
            s = '#' + head + '\n'
            for point in self.pointsInModel:
                s += '{}\t{}\t{}\t{}\n'.format(atom, point[0], point[1], point[2])
            f.write(s)

    def savePDBFile(self, filename='', atom='CA', occupancy=1.0, tempFactor=20.0):
        if filename == '':
            filename = self.modelname + '.pdb'
        with open(filename, 'w') as f:
            s = 'REMARK 265 EXPERIMENT TYPE: THEORETICAL MODELLING\n'
            for i in range(len(self.pointsInModel)):
                x = '{:.2f}'.format(self.pointsInModel[i, 0])
                y = '{:.2f}'.format(self.pointsInModel[i, 1])
                z = '{:.2f}'.format(self.pointsInModel[i, 2])
                s += 'ATOM  {:5d} {:<4} ASP A{:4d}    {:>8}{:>8}{:>8}{:>6}{:>6} 0 2 201\n'.format(int(i), atom, i%10, x, y, z, str(occupancy), str(tempFactor))
            f.write(s)

    def plotSTLMeshModel(self):
        # Create a new plot
        figure = pyplot.figure()
        axes = mplot3d.Axes3D(figure)

        # Load the STL files and add the vectors to the plot
        axes.add_collection3d(mplot3d.art3d.Poly3DCollection(self.stlModelMesh.vectors))
    
        # Auto scale to the mesh size
        scale = self.stlModelMesh.points.flatten(-1)
        axes.auto_scale_xyz(scale, scale, scale)

        # Show the plot to the screen
        pyplot.show()

    def plotPointsInModel(self):
        # Create a new plot
        figure = pyplot.figure()
        axes = mplot3d.Axes3D(figure)

        axes.scatter(self.pointsInModel[:,0], self.pointsInModel[:,1], self.pointsInModel[:,2], color='k')
        # Show the plot to the screen
        pyplot.show()

    def genSasCurve_Crysol(self, qmax=1, qNum=256):
        pdbfile = self.modelname + '.pdb'
        self.savePDBFile(pdbfile)
        os.system('crysol {} -lm 50 -fb 18 -sm {} -ns {} -un 1'.format(pdbfile, qmax, qNum))
        intfile = self.modelname + '00.int'
        crysolOutput = np.loadtxt(intfile, skiprows=1)
        self.sasCurve = crysolOutput[:, :2]
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
        phi = np.arctan2(points_xyz[:,1], points_xyz[:,0]) + np.pi # change from [-pi, pi] to [0, 2pi]
        points_sph = np.vstack((r, theta, phi)).T
        return points_sph  # theta: 0~pi ; phi: 0~2pi

    # unit sphere form factor actually results in wrong outcomes
    # so I delete it ...
    def __Alm(self, q, points_sph, l, m):
        A = 0
        for p_sph in points_sph:
            # p_sph: array[r, theta, phi]; theta: 0~pi, phi: 0~2pi
            r, theta, phi = p_sph[0], p_sph[1], p_sph[2] # theta: 0~pi ; phi: 0~2pi
            A += spherical_jn(l, q*r) * sph_harm(m, l, phi, theta)
        return 4 * np.pi * complex(0,1)**l * A

    # used in func genSasCurve()
    # points in spherical coordinates
    def Iq(self, q, points_sph, lmax):
        I = 0
        for l in range(lmax+1):
            for m in range(-l, l+1):
                I += abs(self.__Alm(q, points_sph, l, m))**2
        return I

    def genSasCurve(self, qmin=0.01, qmax=1, qnum=100, lmax=50):
        self.lmax = lmax
        points_sph = self.xyz2sph(self.pointsInModel)
        q = np.linspace(qmin, qmax, num=qnum)
        
        # this calculation need several minutes in single process mode
        # so use multiprocessing to accelerate
        pool = Pool(self.procNum)
        multip_result_list = []
        length = qnum//self.procNum + 1
        for i in range(self.procNum):
            qslice = q[i*length: (i+1)*length]
            multip_result_list.append(pool.apply_async(self.Iq, args=(qslice, points_sph, lmax,)))
        pool.close()
        pool.join()
        Ilst = []
        for item in multip_result_list:
            Ilst.append(item.get())
        I = np.hstack(Ilst)
        self.sasCurve = np.vstack((q, I)).T
        return self.sasCurve

    def saveSasCurve(self):
        filename = self.modelname + '_saxs.dat'
        header = 'theoretical SAXS curve of {} model\ninterval between points = {}\nl_max in spherical harmonics = {}\n'.format(self.modelname, self.interval, self.lmax)
        header += '\nq\tI'
        np.savetxt(filename, self.sasCurve, header=header)

    def plotSasCurve(self):
        figure = pyplot.figure()
        ax = pyplot.subplot(111)
        ax.set_xscale("log", nonposx='clip')
        ax.set_yscale("log", nonposy='clip')
        pyplot.plot(self.sasCurve[:,0], self.sasCurve[:,1], label=self.modelname)
        pyplot.legend()
        pyplot.show()

if __name__ == '__main__':
    '''
    model = model2sas(procNum=12)
    model.buildFromFile('shell_12_large_hole.xyz')
    #model.savePDBFile()
    #model.plotPointsInModel()
    model.genSasCurve()
    #model.plotSasCurve()
    model.saveSasCurve()
    '''
    model = model2sas(procNum=12)
    boundaryList = [-52, 52, -52, 52, -52, 52]
    model.buildFromMath('cylinder', 'MathDescription', 'cylinder', boundaryList, interval=2)
    model.plotPointsInModel()
    #model.savePointsInModel()
    model.genSasCurve(qnum=200)
    model.plotSasCurve()