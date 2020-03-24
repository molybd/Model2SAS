# -*- coding: utf-8 -*-

import sys, os
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import QFileDialog, QApplication, QMainWindow, QMessageBox

import matplotlib
matplotlib.use("Qt5Agg")  # 声明使用QT5
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from mpl_toolkits import mplot3d

import Model2SAS_UI, Model2SAS
from stl import mesh

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

        # input stl file
        self.ui.pushButton_browseStlFile.clicked.connect(self.browseStlFile)
        
        # set interval and generate points in model
        self.ui.lineEdit_interval.setText('default')
        self.ui.pushButton_genPointsInModel.clicked.connect(self.genPointsModel)

        # save points model
        self.ui.pushButton_savePointsFile.clicked.connect(self.savePointsFile)

        self.ui.checkBox_useCrysol.setChecked(True)

        # choose crysol.exe path
        self.ui.pushButton_chooseCrysolPath.clicked.connect(self.chooseCrysolPath)        
        with open('./CrysolPath.txt', 'r') as f:
            self.crysolPath = f.read()
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
            #self.model.plotSTLMeshModel(plot=True)

            self.plotStlModel()

    def genPointsModel(self):
        # 处理非法输入
        try:
            intervalText = self.ui.lineEdit_interval.text()
            if intervalText == 'default':
                interval = None
            else:
                interval = float(intervalText)
            self.model.buildFromFile(interval=interval)
            #self.model.plotPointsInModel()

            self.plotPointsModel()
        except:
            self.ui.lineEdit_interval.setText('default')

    def savePointsFile(self):
        filetypeText = self.ui.comboBox_choosePointsFileType.currentText()
        filetypeText = filetypeText.lower()
        if 'pdb' in filetypeText:
            self.model.savePointsInModel(filetype='pdb')
        elif 'xyz' in filetypeText:
            self.model.savePointsInModel(filetype='xyz')

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
            self.model.genSasCurve_Crysol(crysolPath=self.crysolPath)
        else:
            self.model.genSasCurve()
        #self.model.plotSasCurve()
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
            self.model.saveSasCurve()
            QMessageBox.information(self.ui,
                "Save successful",
                "SAXS data saved !",
                QMessageBox.Yes
                )
        except:
            print('No SAXS data yet !')
            QMessageBox.warning(self.ui,
                "Save failed",
                "No SAXS data yet !",
                QMessageBox.Yes
                )

    def saveSaxsPlot(self):
        try:
            os.chdir(self.model.inputFileDir)
            filename = self.model.modelname + '_SAXS_Plot.png'
            self.saxsPlotFig.savefig(filename)
            os.chdir(self.model.workingDir)
            QMessageBox.information(self.ui,
                "Save successful",
                "SAXS plot saved !",
                QMessageBox.Yes
                )
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
    
    ui = Model2SAS_UI.Ui_mainWindow()
    ui.setupUi(MainWindow)
    func = function(ui)
    MainWindow.show()
    sys.exit(app.exec_())