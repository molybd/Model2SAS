# -*- coding: utf-8 -*-

# needed for GUI panel
import sys, os, time
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QColor, QStandardItemModel, QStandardItem
from PyQt5.QtWidgets import QFileDialog, QApplication, QMainWindow, QMessageBox, QHeaderView

# needed for multithread
from PyQt5.QtCore import QThread, pyqtSignal

# needed for function
import matplotlib
matplotlib.use("Qt5Agg")  # 声明使用QT5
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from mpl_toolkits import mplot3d

import Model2SAS_UI, Model2SAS
import numpy as np
from stl import mesh
import inspect
from multiprocessing import Process

# 在.ui文件生成的.py文件中，需要把 Ui_MainWindow 类改为 PyQt5.QtWidgets.QWidget 的子类
# 这样才能使用 FileDialog
# 即首先 from PyQt5.QtWidgets import QWidget
# class Ui_MainWindow(Object) 改为 class Ui_MainWindow(QWidget)

class Figure_Canvas(FigureCanvas):   
    # 通过继承FigureCanvas类，使得该类既是一个PyQt5的Qwidget，又是一个matplotlib的FigureCanvas，这是连接pyqt5与matplotlib的关键
    def __init__(self, parent=None, width=5, height=5, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)  # 创建一个Figure，注意：该Figure为matplotlib下的figure，不是matplotlib.pyplot下面的figure
 
        FigureCanvas.__init__(self, self.fig) # 初始化父类
        self.setParent(parent)

        #self.axes = self.fig.add_subplot(111) # 调用figure下面的add_subplot方法，类似于matplotlib.pyplot下面的subplot方法


class Thread_PointsInModel_stl(QThread):

    # 线程结束的signal，并且带有一个列表参数
    threadEnd = pyqtSignal(list)

    def __init__(self, model):
        super(Thread_PointsInModel_stl, self).__init__()
        self.model = model

    def run(self):
        # 线程所需要执行的代码

        self.model.buildFromFile(interval=self.model.interval)

        pointsInModel_list = self.model.pointsInModel.tolist()
        self.threadEnd.emit(pointsInModel_list)


class Thread_PointsInModel_py(QThread):

    # 线程结束的signal，并且带有一个列表参数
    threadEnd = pyqtSignal(list)

    def __init__(self, model, paramsDict):
        super(Thread_PointsInModel_py, self).__init__()
        self.model = model
        self.paramsDict = paramsDict

    def run(self):
        # 线程所需要执行的代码

        # generate custom args string
        arg_string_list = []
        for item in self.paramsDict.items():
            arg_string_list.append('{}={}'.format(item[0], item[1]))
        arg_string = ','.join(arg_string_list)
        exec('self.model.buildFromMath(interval=self.model.interval, useDefault=False, {})'.format(arg_string))
        #self.model.buildFromMath(interval=self.model.interval, useDefault=True)
        
        pointsInModel_list = self.model.pointsInModel.tolist()
        self.threadEnd.emit(pointsInModel_list)


class Thread_Crysol(QThread):
    
    # 线程结束的signal，并且带有一个列表参数
    threadEnd = pyqtSignal(list)

    def __init__(self, model, qmax, lmax, crysolPath):
        super(Thread_Crysol, self).__init__()
        self.model = model
        self.qmax = qmax
        self.lmax = lmax
        self.crysolPath = crysolPath

    def run(self):
        # 线程所需要执行的代码
        sasCurve = self.model.genSasCurve_Crysol(qmax=self.qmax, lmax=self.lmax, crysolPath=self.crysolPath)
        sasCurve_list = sasCurve.tolist()
        self.threadEnd.emit(sasCurve_list)



class function:
    def __init__(self, ui):
        self.ui = ui

        ########### stl file ##############
        # input stl file
        self.ui.pushButton_browseStlFile.clicked.connect(self.browseStlFile)
        ###################################

        ############ py file ##############
        # input py file
        self.ui.pushButton_browsePyFile_math.clicked.connect(self.browsePyFile)
        ###################################

        # set interval and generate points in model
        self.ui.lineEdit_interval.setText('default')
        self.ui.pushButton_genPointsInModel.clicked.connect(self.genPointsModel)

        # set default values of qmax and lmax
        self.ui.lineEdit_Qmax.setText('1')
        self.ui.lineEdit_lmax.setText('20')

        # save points model
        self.ui.pushButton_savePointsFile.clicked.connect(self.savePointsFile)

        self.ui.checkBox_useCrysol.setChecked(True)

        # choose crysol.exe path
        self.ui.pushButton_chooseCrysolPath.clicked.connect(self.chooseCrysolPath)  
        try:      
            with open('./CrysolPath.txt', 'r') as f:
                self.crysolPath = f.read()
        except:
            with open('./CrysolPath.txt', 'w') as f:
                f.write('')
            self.crysolPath = 'Please choose crysol path'
        self.ui.label_crysolPath.setText(self.crysolPath)

        # generate SAXS curve
        self.ui.pushButton_calcSaxs.clicked.connect(self.genSaxsCurve)

        # save SAXS data
        self.ui.pushButton_saveSaxsData.clicked.connect(self.saveSaxsData)

        # save SAXS plot
        self.ui.pushButton_saveSaxsPlot.clicked.connect(self.saveSaxsPlot)

        self.needGenSaxsCurve = False


    def browseStlFile(self):
        #options = QFileDialog.Options()
        #options |= QFileDialog.DontUseNativeDialog
        file, _ = QFileDialog.getOpenFileName(self.ui, "select stl file", "","All Files (*);;stl Files (*.stl)")
        # here, seperator in file is "/"
        if file:
            self.stlFile = file
            print(file)
            self.ui.lineEdit_stlFile.setText(self.stlFile)
            self.model = Model2SAS.model2sas(self.stlFile, autoGenPoints=False)
            self.model.stlModelMesh = mesh.Mesh.from_file(self.stlFile)

            self.plotStlModel()

            # determine default interval
            vectors = self.model.stlModelMesh.vectors
            xmin, xmax, ymin, ymax, zmin, zmax = np.min(vectors[:,:,0]), np.max(vectors[:,:,0]), np.min(vectors[:,:,1]), np.max(vectors[:,:,1]), np.min(vectors[:,:,2]), np.max(vectors[:,:,2])
            interval = min([xmax-xmin, ymax-ymin, zmax-zmin]) / 20
            if interval < 0.5:
                interval = 0.5
            interval = float('{:.2f}'.format(interval))  # 只保留两位小数
            self.defaultInterval = interval
            self.model.interval = interval
            self.ui.lineEdit_interval.setText('{:.2f}'.format(interval))

    def browsePyFile(self):
        #options = QFileDialog.Options()
        #options |= QFileDialog.DontUseNativeDialog
        file, _ = QFileDialog.getOpenFileName(self.ui, "select py file", "","All Files (*);;stl Files (*.py)")
        # here, seperator in file is "/"
        if file:
            self.pyFile = file
            print(file)
            self.ui.lineEdit_pyFile_math.setText(self.pyFile)
            self.model = Model2SAS.model2sas(self.pyFile, autoGenPoints=False)

            self.showParamsTable()
            self.showPyFileCode()
            
            # ! determine default interval in self.showParamsTable()

    
    def showParamsTable(self):
        sys.path.append(self.model.inputFileDir)   # add dir to sys.path, then we can directly import the py file
        modelModule = '.'.join(self.model.inputFileName.split('.')[:-1])
        mathModel = __import__(modelModule)
        # read function arguments
        # inspect.getargspec() is deprecated since Python 3.0
        # args = inspect.getargspec(mathModel.model) 
        sig = inspect.signature(mathModel.model)
        params = sig.parameters
        self.mathParams = params

        rowNum = len(params) - 2
        self.rowNum = rowNum
        colNum = 2
        self.tableModel = QStandardItemModel(rowNum, colNum)
        self.tableModel.setHorizontalHeaderLabels(['parameter', 'value'])

        keys = list(params.keys())
        for row in range(rowNum):
            paramName = keys[row+1]
            paramValue = params[paramName].default
            i1 = QStandardItem(str(paramName))
            self.tableModel.setItem(row, 0, i1)
            i2 = QStandardItem(str(paramValue))
            self.tableModel.setItem(row, 1, i2)
        self.ui.tableView_params_math.setModel(self.tableModel)
        # 设置行列填满窗口
        self.ui.tableView_params_math.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.ui.tableView_params_math.verticalHeader().setSectionResizeMode(QHeaderView.Stretch)

        # set default interval
        boundaryList = params['boundary_xyz'].default
        [xmin, xmax, ymin, ymax, zmin, zmax] = boundaryList
        interval = min(np.abs([xmax-xmin, ymax-ymin, zmax-zmin])) / 20
        if interval < 0.5:
            interval = 0.5
        interval = float('{:.2f}'.format(interval))  # 只保留一位小数
        self.defaultInterval = interval
        self.model.interval = interval
        self.ui.lineEdit_interval.setText('{:.2f}'.format(interval))



    # get params table content
    # to deal with the case that params values need to be modiied
    def readParamsTable(self):
        paramsDict = {}
        for row in range(self.rowNum):
            paramName = self.ui.tableView_params_math.model().index(row, 0).data()
            paramValue = self.ui.tableView_params_math.model().index(row, 1).data()
            paramsDict[paramName] = paramValue
        print(paramsDict)
        self.paramsDict = paramsDict
        return paramsDict

    def showPyFileCode(self):
        self.ui.textBrowser_pyFileContent.clear()
        
        with open(self.pyFile, 'r') as f:
            pyFileContentList = f.readlines() 

        for line in pyFileContentList:
            self.ui.textBrowser_pyFileContent.append(line)
        self.ui.textBrowser_pyFileContent.moveCursor(self.ui.textBrowser_pyFileContent.textCursor().End)


    def genPointsModel(self):
        # 结束前禁用这个 button，以免被疯狂点击
        self.ui.pushButton_genPointsInModel.setEnabled(False)

        # progress bar 开始滚动
        self.ui.progressBar.setMinimum(0)
        self.ui.progressBar.setMaximum(0)

        # 处理interval的非法输入并确定interval
        try:
            intervalText = self.ui.lineEdit_interval.text()
            interval = float(intervalText)
        except:
            interval = self.defaultInterval
        self.ui.lineEdit_interval.setText('{:.2f}'.format(interval))
        self.model.interval = interval

        if self.model.inputFileType == 'stl':
            self.thread_pointsInModel_stl = Thread_PointsInModel_stl(self.model)
            self.thread_pointsInModel_stl.threadEnd.connect(self.processPointsInModelThreadOutput)
            self.thread_pointsInModel_stl.start()
        if self.model.inputFileType == 'py':
            paramsDict = self.readParamsTable()
            self.thread_pointsInModel_py = Thread_PointsInModel_py(self.model, paramsDict)
            self.thread_pointsInModel_py.threadEnd.connect(self.processPointsInModelThreadOutput)
            self.thread_pointsInModel_py.start()
            

    def processPointsInModelThreadOutput(self, pointsInModel_list):
        self.model.pointsInModel = np.array(pointsInModel_list)

        self.ui.lineEdit_interval.setText('{:.2f}'.format(self.model.interval))
        pointsNum = self.model.pointsInModel.shape[0]
        self.ui.label_pointsNum.setText('{} points'.format(pointsNum))

        self.plotPointsModel()

        # progress bar 停止滚动
        self.ui.progressBar.setMinimum(0)
        self.ui.progressBar.setMaximum(100)
        # button 恢复
        self.ui.pushButton_genPointsInModel.setEnabled(True)

        if self.needGenSaxsCurve:
            self._onlyGenSaxsCurve()

        self.needGenSaxsCurve = False

    def savePointsFile(self):
        try:
            basename = self.model.modelname + '.pdb'
            file, _ = QFileDialog.getSaveFileName(self.ui, 'save PDB file', basename)
            if file:
                self.model.savePointsInModel(filetype='pdb', filename=file)
        except:
            print('No points data yet !')
            QMessageBox.warning(self.ui,
                "Save failed",
                "No points data yet !",
                QMessageBox.Yes
                )

    def chooseCrysolPath(self):
        file, _ = QFileDialog.getOpenFileName(self.ui, "select crysol.exe file", "","All Files (*);;exe Files (*.exe)")
        # here, seperator in file is "/"
        self.crysolPath = file
        self.ui.label_crysolPath.setText(self.crysolPath)
        with open('./CrysolPath.txt', 'w') as f:
            f.write(self.crysolPath)

    def genSaxsCurve(self):
        # 结束前禁用这个 button，以免被疯狂点击
        self.ui.pushButton_calcSaxs.setEnabled(False)

        if self.model.pointsInModel.size == 0 or abs(self.model.interval - float(self.ui.lineEdit_interval.text())) >= 0.05:
            # in case that pointsInModel haven't been generated
            print('new points model')
            self.needGenSaxsCurve = True
            self.genPointsModel()
        
        else:
            # model have already been built

            self._onlyGenSaxsCurve()

    def _onlyGenSaxsCurve(self):

        # progress bar 开始滚动
        self.ui.progressBar.setMinimum(0)
        self.ui.progressBar.setMaximum(0)

        useCrysol = self.ui.checkBox_useCrysol.isChecked()
        if useCrysol:
            qmax = float(self.ui.lineEdit_Qmax.text())
            lmax = int(self.ui.lineEdit_lmax.text())
            self.model.lmax = lmax

            self.thread_crysol = Thread_Crysol(self.model, qmax, lmax, self.crysolPath)
            self.thread_crysol.threadEnd.connect(self.processCrysolThreadOutput)
            self.thread_crysol.start()
            
            # self.model.genSasCurve_Crysol(crysolPath=self.crysolPath, qmax=qmax, lmax=lmax)
        else:
            self.model.genSasCurve(qmax=qmax, lmax=lmax)

        # self.plotSasCurve()

    def processCrysolThreadOutput(self, sasCurve_list):
        self.model.sasCurve = np.array(sasCurve_list)
        self.plotSasCurve()

        # progress bar 停止滚动
        self.ui.progressBar.setMinimum(0)
        self.ui.progressBar.setMaximum(100)
        # button 恢复
        self.ui.pushButton_calcSaxs.setEnabled(True)

    def plotStlModel(self):
        stlPlot = Figure_Canvas()
        stlPlotAxes = mplot3d.Axes3D(stlPlot.fig)
        # Load the STL files and add the vectors to the plot

        # plot model face
        '''
        Poly3DCollection = mplot3d.art3d.Poly3DCollection(
            self.model.stlModelMesh.vectors, 
            facecolors='w',
            #linewidths=1, 
            alpha=0
            )
        stlPlotAxes.add_collection3d(Poly3DCollection)
        '''
        # plot model frame mesh
        Line3DCollection = mplot3d.art3d.Line3DCollection(
            self.model.stlModelMesh.vectors, 
            colors='k', 
            linewidths=0.5, 
            #linestyles=':'
            )
        stlPlotAxes.add_collection3d(Line3DCollection)
        '''
        # Auto scale to the mesh size
        scale = self.model.stlModelMesh.points.flatten()
        stlPlotAxes.auto_scale_xyz(scale, scale, scale)
        '''
        # scale the 3 axis, make them the same range
        xmin, xmax = self.model.stlModelMesh.vectors[:,:,0].min(), self.model.stlModelMesh.vectors[:,:,0].max()
        ymin, ymax = self.model.stlModelMesh.vectors[:,:,1].min(), self.model.stlModelMesh.vectors[:,:,1].max()
        zmin, zmax = self.model.stlModelMesh.vectors[:,:,2].min(), self.model.stlModelMesh.vectors[:,:,2].max()
        dmax = max([xmax - xmin, ymax - ymin, zmax - zmin])
        xmid, ymid, zmid = (xmin + xmax)/2, (ymin + ymax)/2, (zmin + zmax)/2
        xmin, xmax = xmid - 0.6*dmax, xmid + 0.6*dmax
        ymin, ymax = ymid - 0.6*dmax, ymid + 0.6*dmax
        zmin, zmax = zmid - 0.6*dmax, zmid + 0.6*dmax
        stlPlotAxes.set_xlim3d(xmin, xmax)
        stlPlotAxes.set_ylim3d(ymin, ymax)
        stlPlotAxes.set_zlim3d(zmin, zmax)
        
        '''
        # frame plot
        # too slow ...
        for triangle in self.model.stlModelMesh.vectors:
            x, y, z = triangle[:,0], triangle[:,1], triangle[:,2]
            x = np.hstack((x, x[0]))
            y = np.hstack((y, y[0]))
            z = np.hstack((z, z[0]))
            stlPlotAxes.plot(x, y, z, color='b')
        '''


        # 创建一个QGraphicsScene，因为加载的图形（FigureCanvas）不能直接放到graphicview控件中，必须先放到graphicScene，然后再把graphicscene放到graphicview中
        stlGraphicscene = QtWidgets.QGraphicsScene()
        # 把图形放到QGraphicsScene中，注意：图形是作为一个QWidget放到QGraphicsScene中的
        stlGraphicscene.addWidget(stlPlot)
        self.ui.graphicsView_stlModel.setScene(stlGraphicscene)
        self.ui.graphicsView_stlModel.show()

    def plotPointsModel(self):
        pointsPlot = Figure_Canvas()
        pointsPlotAxes = mplot3d.Axes3D(pointsPlot.fig)
        pointsPlotAxes.scatter(
            self.model.pointsInModel[:,0], 
            self.model.pointsInModel[:,1], 
            self.model.pointsInModel[:,2], 
            color='k')
        
        # set the scale of plot, all axes have the same range
        xmin, xmax = self.model.pointsInModel[:,0].min(), self.model.pointsInModel[:,0].max()
        ymin, ymax = self.model.pointsInModel[:,1].min(), self.model.pointsInModel[:,1].max()
        zmin, zmax = self.model.pointsInModel[:,2].min(), self.model.pointsInModel[:,2].max()
        dmax = max([xmax - xmin, ymax - ymin, zmax - zmin])
        xmid, ymid, zmid = (xmin + xmax)/2, (ymin + ymax)/2, (zmin + zmax)/2
        xmin, xmax = xmid - 0.6*dmax, xmid + 0.6*dmax
        ymin, ymax = ymid - 0.6*dmax, ymid + 0.6*dmax
        zmin, zmax = zmid - 0.6*dmax, zmid + 0.6*dmax
        pointsPlotAxes.set_xlim3d(xmin, xmax)
        pointsPlotAxes.set_ylim3d(ymin, ymax)
        pointsPlotAxes.set_zlim3d(zmin, zmax)

        # 创建一个QGraphicsScene，因为加载的图形（FigureCanvas）不能直接放到graphicview控件中，必须先放到graphicScene，然后再把graphicscene放到graphicview中
        pointsGraphicscene = QtWidgets.QGraphicsScene()
        # 把图形放到QGraphicsScene中，注意：图形是作为一个QWidget放到QGraphicsScene中的
        pointsGraphicscene.addWidget(pointsPlot)
        self.ui.graphicsView_pointsModel.setScene(pointsGraphicscene)
        self.ui.graphicsView_pointsModel.show()

    def plotSasCurve(self, show=True, save=False):
        saxsPlot = Figure_Canvas(width=7.5, height=5, dpi=100)
        self.saxsPlotFig = saxsPlot.fig
        saxsPlotAxes = saxsPlot.fig.add_subplot(111)

        saxsPlotAxes.plot(
            self.model.sasCurve[:,0],
            self.model.sasCurve[:,1],
            '-', label=self.model.modelname
            )

        saxsPlotAxes.set_xscale('log')
        saxsPlotAxes.set_yscale('log')

        saxsPlotAxes.set_xlabel(r'Q $(\AA^{-1})$', fontsize=13)
        saxsPlotAxes.set_ylabel(r'Intensity (a.u.)', fontsize=13)

        saxsPlotAxes.legend(fontsize=11, frameon=False)

        # 创建一个QGraphicsScene，因为加载的图形（FigureCanvas）不能直接放到graphicview控件中，必须先放到graphicScene，然后再把graphicscene放到graphicview中
        saxsGraphicscene = QtWidgets.QGraphicsScene()
        # 把图形放到QGraphicsScene中，注意：图形是作为一个QWidget放到QGraphicsScene中的
        saxsGraphicscene.addWidget(saxsPlot)
        self.ui.graphicsView_saxsPlot.setScene(saxsGraphicscene)
        self.ui.graphicsView_saxsPlot.show()

    def saveSaxsData(self):
        try:
            basename = self.model.modelname + '_saxs.dat'
            file, _ = QFileDialog.getSaveFileName(self.ui, 'Save SAXS Data', basename)
            if file:
                self.model.saveSasCurve(filename=file)
        except:
            print('No SAXS data yet !')
            QMessageBox.warning(self.ui,
                "Save failed",
                "No SAXS data yet !",
                QMessageBox.Yes
                )

    def saveSaxsPlot(self):
        try:
            basename = self.model.modelname + '_SAXS_Plot.png'
            file, _ = QFileDialog.getSaveFileName(self.ui, 'Save SAXS Plot', basename)
            if file:
                self.saxsPlotFig.savefig(file)
        except:
            print('No SAXS plot yet !')
            QMessageBox.warning(self.ui,
                "Save failed",
                "No SAXS plot yet !",
                QMessageBox.Yes
                )

    


if __name__ == "__main__":
    app = QApplication(sys.argv)
    MainWindow = QMainWindow()
    
    ui = Model2SAS_UI.Ui_MainWindow()
    ui.setupUi(MainWindow)
    func = function(ui)
    MainWindow.show()
    sys.exit(app.exec_())