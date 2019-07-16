# -*- coding: UTF-8 -*-

import numpy as np
from stl import mesh
from mpl_toolkits import mplot3d
from matplotlib import pyplot
from scipy.special import sph_harm, spherical_jn, jv
import os
from multiprocessing import Pool

class model2sas:
    'class to read 3D model from file and generate PDB file and SAS curve'

    def __init__(self, multiproc=False, procNum=1, *args, **kwargs):
        self.modelname = ''
        self.meshgrid = np.array([])
        self.pointsInModel = np.array([])
        self.stlModelMesh = None
        self.sasCurve = np.array([])
        self.multiproc = multiproc
        self.procNum = 1
        if self.multiproc:
            self.procNum = procNum
        else:
            self.procNum = 1

    def generateMeshgrid(self, xmin, xmax, ymin, ymax, zmin, zmax, interval=1):
        xscale = np.linspace(xmin, xmax, num=int((xmax-xmin)/interval+1))
        yscale = np.linspace(ymin, ymax, num=int((ymax-ymin)/interval+1))
        zscale = np.linspace(zmin, zmax, num=int((zmax-zmin)/interval+1))
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

    # for the usage of multiprocessing
    def ptsInSTLModel(self, pts):
        ptsInModelList = []
        for pt in pts:
            if self.isPointInSTLModel(pt, self.stlModelMesh)[0]:
                ptsInModelList.append(pt)
        return ptsInModelList

    # to generate a 3D model from stl file
    def build_from_STLfile(self, stlfile, interval=1, modelname=''):
        self.interval = interval
        if modelname == '':
            if '/' in stlfile:
                self.modelname = stlfile.split('/')[-1].split('.')[-2]
            elif '\\' in stlfile:
                self.modelname = stlfile.split('\\')[-1].split('.')[-2]
            else:
                self.modelname = stlfile.split('.')[-2]
        else:
            self.modelname = modelname
            
        self.stlModelMesh = mesh.Mesh.from_file(stlfile)
        vectors = self.stlModelMesh.vectors
        xmin, xmax, ymin, ymax, zmin, zmax = np.min(vectors[:,:,0]), np.max(vectors[:,:,0]), np.min(vectors[:,:,1]), np.max(vectors[:,:,1]), np.min(vectors[:,:,2]), np.max(vectors[:,:,2])
        self.generateMeshgrid(xmin, xmax, ymin, ymax, zmin, zmax, interval=interval)

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
        return self.pointsInModel
    
    # to generate a 3D model from a mathematical description
    # for example: a hollow sphere is "x**2+y**2+z**2 >= R1**2 and x**2+y**2+z**2 <= R2**2
    # this description must be a python boolean expression !
    # the coordinate of point must be x,y,z
    # boundaryList is [xmin, xmax, ymin, ymax, zmin, zmax]
    def build_from_MathDescription(self, modelname, description, boundaryList, interval=1):
        self.modelname = modelname
        xmin, xmax, ymin, ymax, zmin, zmax = boundaryList[0], boundaryList[1], boundaryList[2], boundaryList[3], boundaryList[4], boundaryList[5]
        self.generateMeshgrid(xmin, xmax, ymin, ymax, zmin, zmax, interval=interval)
        pointsInModelList = []
        for point in self.meshgrid:
            x, y, z = point[0], point[1], point[2]
            if eval(description):
                pointsInModelList.append(point)
        self.pointsInModel = np.array(pointsInModelList)
        return self.pointsInModel

    def build_from_PointsFile(self, filename, interval=1, modelname=''):
        self.interval = interval
        if modelname == '':
            if '/' in filename:
                self.modelname = filename.split('/')[-1].split('.')[-2]
            elif '\\' in filename:
                self.modelname = filename.split('\\')[-1].split('.')[-2]
            else:
                self.modelname = filename.split('.')[-2]
        else:
            self.modelname = modelname
        self.pointsInModel = np.loadtxt(filename)
        return self.pointsInModel

    def build_from_XYZfile(self, filename, interval=1, modelname=''):
        self.interval = interval
        if modelname == '':
            if '/' in filename:
                self.modelname = filename.split('/')[-1].split('.')[-2]
            elif '\\' in filename:
                self.modelname = filename.split('\\')[-1].split('.')[-2]
            else:
                self.modelname = filename.split('.')[-2]
        else:
            self.modelname = modelname
        
        lst = []
        with open(filename, 'r') as f:
            for line in f.readlines():
                if line[0] != '#':
                    point = [float(i) for i in line.split()[1:4]]
                    lst.append(point)
        self.pointsInModel = np.array(lst)
        return self.pointsInModel


    def writePointsInModel(self):
        filename = '{}_interval={}.txt'.format(self.modelname, self.interval)
        np.savetxt(filename, self.pointsInModel)

    def writeXYZfile(self, filename='', head='created by limu', atom='CA'):
        if filename == '':
            filename = self.modelname + '.xyz'
        with open(filename, 'w') as f:
            s = '#' + head + '\n'
            for point in self.pointsInModel:
                s += '{}\t{}\t{}\t{}\n'.format(atom, point[0], point[1], point[2])
            f.write(s)

    def writePDBfile(self, filename='', atom='CA', occupancy=1.0, tempFactor=20.0):
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
        self.writePDBfile(pdbfile)
        os.system('crysol {} -lm 50 -fb 18 -sm {} -ns {} -un 1'.format(pdbfile, qmax, qNum))
        intfile = self.modelname + '00.int'
        crysolOutput = np.loadtxt(intfile, skiprows=1)
        self.sasCurve = crysolOutput[:, :2]
        return self.sasCurve

    def xyz2sph(self, point):
        epsilon=1e-100
        r = np.linalg.norm(point)
        theta = np.arccos(point[2] / (r+epsilon))
        phi = np.arctan(point[1] / (point[0]+epsilon))
        return np.array([r, theta, phi]) # theta: 0~pi ; phi: 0~2pi

    def Alm(self, q, points, l, m, unitFormFactor=True):
        A = 0
        if unitFormFactor == True:
            R = self.interval / 2
            for p in points:
                p_sph = self.xyz2sph(p) # p_sph: array[r, theta, phi]; theta: 0~pi, phi: 0~2pi
                r, theta, phi = p_sph[0], p_sph[1], p_sph[2] # theta: 0~pi ; phi: 0~2pi
                A += (jv(1.5, q*R)/(q*R)**1.5) * spherical_jn(l, q*r) * sph_harm(m, l, phi, theta)
        else:
            for p in points:
                p_sph = self.xyz2sph(p) # p_sph: array[r, theta, phi]; theta: 0~pi, phi: 0~2pi
                r, theta, phi = p_sph[0], p_sph[1], p_sph[2] # theta: 0~pi ; phi: 0~2pi
                A += spherical_jn(l, q*r) * sph_harm(m, l, phi, theta)
        return 4 * np.pi * complex(0,1)**l * A

    def Iq(self, q, points, lmax=5, unitFormFactor=True):
        I = 0
        for l in range(lmax+1):
            for m in range(-l, l+1):
                I += abs(self.Alm(q, points, l, m, unitFormFactor=True))**2
        return I

    def genSasCurve(self, qmin=0.01, qmax=1, qnum=50, lmax=10, unitFormFactor=True):
        q = np.linspace(qmin, qmax, num=qnum)
        I = self.Iq(q, self.pointsInModel, lmax=lmax, unitFormFactor=unitFormFactor)
        self.sasCurve = np.vstack((q, I)).T
        return self.sasCurve

    def saveSasCurve(self):
        filename = self.modelname + '_saxs.dat'
        header = 'theoretical SAXS curve of {} model\n'.format(self.modelname)
        header += 'q\tI'
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
    torus1 = model2sas(multiproc=True, procNum=8)
    torus1.build_from_STLfile('torus.stl')
    torus1.plotPointsInModel()