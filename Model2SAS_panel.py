# -*- coding: utf-8 -*-

import sys, os, time
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QColor, QStandardItemModel, QStandardItem
from PyQt5.QtWidgets import QFileDialog, QApplication, QMainWindow, QMessageBox, QHeaderView

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
            
            self.ui.lineEdit_interval.setText('default')

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

            self.ui.lineEdit_interval.setText('default')
    
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
        # 处理interval的非法输入
        try:
            intervalText = self.ui.lineEdit_interval.text()
            if intervalText == 'default':
                interval = None
            else:
                interval = float(intervalText)
        except:
            self.ui.lineEdit_interval.setText('default')
            interval = None
        self.model.interval = interval
        
        if self.model.inputFileType == 'stl':
            self.model.buildFromFile(interval=self.model.interval)

        elif self.model.inputFileType == 'py':
            self.readParamsTable()

            # generate custom args string
            arg_string_list = []
            for item in self.paramsDict.items():
                arg_string_list.append('{}={}'.format(item[0], item[1]))
            arg_string = ','.join(arg_string_list)
            exec('self.model.buildFromMath(interval=self.model.interval, useDefault=False, {})'.format(arg_string))
            #self.model.buildFromMath(interval=self.model.interval, useDefault=True)

        self.ui.lineEdit_interval.setText('{:.1f}'.format(self.model.interval))
        pointsNum = self.model.pointsInModel.shape[0]
        self.ui.label_pointsNum.setText('{} points'.format(pointsNum))

        self.plotPointsModel()

    def savePointsFile(self):
        basename = self.model.modelname + '.pdb'
        file, _ = QFileDialog.getSaveFileName(self.ui, 'save PDB file', basename)
        self.model.savePointsInModel(filetype='pdb', filename=file)

    def chooseCrysolPath(self):
        file, _ = QFileDialog.getOpenFileName(self.ui, "select crysol.exe file", "","All Files (*);;exe Files (*.exe)")
        # here, seperator in file is "/"
        self.crysolPath = file
        self.ui.label_crysolPath.setText(self.crysolPath)
        with open('./CrysolPath.txt', 'w') as f:
            f.write(self.crysolPath)

    def genSaxsCurve(self):
        # incase that pointsInModel haven't been generated
        if len(self.model.pointsInModel) == 0:
            self.genPointsModel()
        
        useCrysol = self.ui.checkBox_useCrysol.isChecked()
        if useCrysol:
            qmax = float(self.ui.lineEdit_Qmax.text())
            lmax = int(self.ui.lineEdit_lmax.text())
            self.model.genSasCurve_Crysol(crysolPath=self.crysolPath, qmax=qmax, lmax=lmax)
        else:
            self.model.genSasCurve(qmax=qmax, lmax=lmax)

        self.plotSasCurve()

    def plotStlModel(self):
        stlPlot = Figure_Canvas()
        stlPlotAxes = mplot3d.Axes3D(stlPlot.fig)
        # Load the STL files and add the vectors to the plot
        stlPlotAxes.add_collection3d(mplot3d.art3d.Poly3DCollection(self.model.stlModelMesh.vectors))
        # Auto scale to the mesh size
        scale = self.model.stlModelMesh.points.flatten()
        stlPlotAxes.auto_scale_xyz(scale, scale, scale)

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
        min_all = self.model.meshgrid.min()
        max_all = self.model.meshgrid.max()
        pointsPlotAxes.set_xlim3d(min_all, max_all)
        pointsPlotAxes.set_ylim3d(min_all, max_all)
        pointsPlotAxes.set_zlim3d(min_all, max_all)

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
            self.model.saveSasCurve(filename=file)
            '''
            QMessageBox.information(self.ui,
                "Save successful",
                "SAXS data saved !",
                QMessageBox.Yes
                )'''
        except:
            print('No SAXS data yet !')
            QMessageBox.warning(self.ui,
                "Save failed",
                "No SAXS data saved yet !",
                QMessageBox.Yes
                )

    def saveSaxsPlot(self):
        try:
            basename = self.model.modelname + '_SAXS_Plot.png'
            file, _ = QFileDialog.getSaveFileName(self.ui, 'Save SAXS Plot', basename)
            self.saxsPlotFig.savefig(file)
            '''
            QMessageBox.information(self.ui,
                "Save successful",
                "SAXS plot saved !",
                QMessageBox.Yes
                )'''
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