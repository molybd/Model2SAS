# -*- coding: UTF-8 -*-

import os
import time
import numpy as np
from stl import mesh
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

from Model2SAS import *
from Plot import *
from Functions import intensity_parallel, intensity

# 以下均为GUI相关的导入
import sys
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import  QWidget, QApplication, QMainWindow, QMdiSubWindow, QFileDialog, QDialog, QInputDialog, QHeaderView, QAbstractItemView
from PyQt5.QtGui import QStandardItemModel, QStandardItem

# needed for plot
import matplotlib
matplotlib.use("Qt5Agg")  # 声明使用QT5
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# my own qtgui files
from qtgui.mainwindow_ui import Ui_mainWindow
from qtgui.stlmodelView_ui import Ui_stlmodelView
from qtgui.mathmodelView_ui import Ui_mathmodelView
from qtgui.sasdataView_ui import Ui_sasdataView
from qtgui.pointsWithSldView_ui import Ui_pointsWithSldView

# needed for multithread
from PyQt5.QtCore import QThread, pyqtSignal


''' 尚待解决的问题
Bug:
1. 模型点数太多的时候有时会报错
2. 计算SAS曲线时q点数太少会报错
主要功能：
(solved) 1. 改变stl模型sld的功能
(shelved) 2. 改变math模型参数的功能
次要功能：
(solved) 1. 删除模型
(solved) 2. 删除所有模型
3. 保存project
程序结构：
(solved) 1. genPoints() 异步进行
(solved) 2. control panel 变成dock widget
'''


class stlmodelViewWindow(QWidget, Ui_stlmodelView):
    def __init__(self, parent=None):
        super(stlmodelViewWindow, self).__init__(parent)
        self.setupUi(self)
class mathmodelViewWindow(QWidget, Ui_mathmodelView):
    def __init__(self, parent=None):
        super(mathmodelViewWindow, self).__init__(parent)
        self.setupUi(self)
class pointsWithSldViewWindow(QWidget, Ui_pointsWithSldView):
    def __init__(self, parent=None):
        super(pointsWithSldViewWindow, self).__init__(parent)
        self.setupUi(self)
class sasdataViewWindow(QWidget, Ui_sasdataView):
    def __init__(self, parent=None):
        super(sasdataViewWindow, self).__init__(parent)
        self.setupUi(self)


# 通过继承FigureCanvas类，使得该类既是一个PyQt5的Qwidget，又是一个matplotlib的FigureCanvas，这是连接pyqt5与matplotlib的关键！
# 这样就可以把 matplotlib 画的图嵌入到pyqt的GUI窗口中了
# 并且可以实现画的三维图可动
class Figure_Canvas(FigureCanvas):
    '''Usage
    canvas = Figure_Canvas(figsize=(8,4))
    plotStlMeshes(mesh_list, show=False, figure=canvas.figure)
    # 创建一个QGraphicsScene，因为加载的图形（FigureCanvas）不能直接放到graphicview控件中，必须先放到graphicScene，然后再把graphicscene放到graphicview中
    graphicScene = QtWidgets.QGraphicsScene()
    # 把图形放到QGraphicsScene中，注意：图形是作为一个QWidget放到QGraphicsScene中的
    graphicScene.addWidget(canvas)
    stlmodelView.graphicsView.setScene(graphicScene)
    '''
    def __init__(self, parent=None, **kwargs):
        self.figure = Figure(**kwargs)  # 创建一个Figure，注意：该Figure为matplotlib下的figure，不是matplotlib.pyplot下面的figure
        FigureCanvas.__init__(self, self.figure) # 初始化父类
        self.setParent(parent)

class EmittingStream(QtCore.QObject):
    '''写一个信号，用来发射标准输出作为信号，为了在console中显示print的值和错误信息
    '''
    textWritten = QtCore.pyqtSignal(str)  #定义一个发送str的信号
    def write(self, text):
        self.textWritten.emit(str(text))  

class Thread_calcSas(QThread):
    # 线程结束的signal，并且带有一个ndarray参数
    threadEnd = pyqtSignal(np.ndarray)
    def __init__(self, q, points, sld, lmax, parallel, cpu_usage, proc_num):
        super(Thread_calcSas, self).__init__()
        self.q = q
        self.points = points
        self.sld = sld
        self.lmax = lmax
        self.parallel = parallel
        self.cpu_usage = cpu_usage
        self.proc_num = proc_num
    def run(self):
        # 线程所需要执行的代码
        if self.parallel:
            self.I = intensity_parallel(self.q, self.points, self.sld, self.lmax, cpu_usage=self.cpu_usage, proc_num=self.proc_num)
        else:
            self.I = intensity(self.q, self.points, self.sld, self.lmax)
        self.threadEnd.emit(self.I)

class Thread_genPoints(QThread):
    threadEnd = pyqtSignal(object)
    def __init__(self, project, grid_num=10000, interval=None):
        super(Thread_genPoints, self).__init__()
        self.temp_project = project
        self.grid_num = grid_num
        self.interval = interval
    def run(self):
        if self.interval:
            self.temp_project.genPoints(interval=self.interval)
        elif self.grid_num:
            self.temp_project.genPoints(grid_num=self.grid_num)
        self.threadEnd.emit(self.temp_project)
        



class mainwindowFunction:
    
    def __init__(self, ui):
        default_name = 'New Project'
        project = model2sas(default_name)
        self.project = project

        self.ui = ui
        self.ui.actionNew_Project.triggered.connect(self.newProject)
        self.ui.actionImport_model_s.triggered.connect(self.importModels)
        self.ui.actionDelete_all_models.triggered.connect(self.deleteAllModels)
        self.ui.actionCascade_2.triggered.connect(self.ui.mdiArea.cascadeSubWindows)
        self.ui.actionTile_2.triggered.connect(self.ui.mdiArea.tileSubWindows)
        
        self.ui.label_projectName.setText('Project: {}'.format(self.project.name))
        self.ui.pushButton_importModels.clicked.connect(self.importModels)
        self.ui.pushButton_deleteModel.clicked.connect(self.deleteModel)
        self.ui.pushButton_showStlmodels.clicked.connect(self.showStlModels)
        self.ui.pushButton_showMathmodel.clicked.connect(self.showMathModel)
        self.ui.pushButton_genPoints.clicked.connect(self.genPoints)
        self.ui.pushButton_calcSas.clicked.connect(self.calcSas)
        
        #下面将输出重定向到textEdit中
        #sys.stdout = EmittingStream(textWritten=self.outputWritten) 
        #sys.stderr = EmittingStream(textWritten=self.outputWritten)
        
        self.refreshTableViews()
        self.consolePrint('New project established with name: {}'.format(self.project.name))

    #接收信号str的信号槽
    def outputWritten(self, text):
        cursor = self.ui.textEdit.textCursor()
        cursor.movePosition(QtGui.QTextCursor.End)  
        cursor.insertText(text)
        self.ui.textEdit.setTextCursor(cursor)
        self.ui.textEdit.ensureCursorVisible()

    def newProject(self):
        name, ok_pressed = QInputDialog.getText(None, 'New Project', 'Name: ')
        if ok_pressed:
            if name == '':
                name = 'New Project'
            project = model2sas(name)
            self.project = project
            self.ui.label_projectName.setText('Project: {}'.format(self.project.name))
            self.consolePrint('New project established with name: {}'.format(self.project.name))
    
    def deleteAllModels(self):
        name = self.project.name
        project = model2sas(name)
        self.project = project
        self.refreshTableViews()

    def importModels(self):
        filepath_list, filetype_list = QFileDialog.getOpenFileNames(None, 'Select Model File(s)', './', "All Files (*);;stl Files (*.stl);;math model Files (*.py)")
        '''
        ###### TEST ######
        filepath_list = ['models\shell_12hole.STL', 'models\\torus.STL', 'models\\new_hollow_sphere_model.py']
        project = model2sas('test', 'models/projects')
        project.setupModel()
        self.project = project
        self.ui.label_projectName.setText(self.project.name)
        ##################
        '''
        for filepath in filepath_list:
            self.project.importFile(filepath, sld=1)
            filetype = filepath.split('.')[-1].upper()
            if filetype == 'STL':
                self.consolePrint('Import {} models with path: {}'.format(filetype, filepath))
            elif filetype == 'PY':
                self.consolePrint('Import {} models with path: {}'.format('MATH', filepath))
        self.refreshTableViews()

    def deleteModel(self):
        # delete selected stlmodel
        indexes = self.ui.tableView_stlmodels.selectionModel().selectedRows()
        if len(indexes) != 0:
            i = indexes[0].row()  # 只删除选中的第一行
            self.consolePrint('Delete STL model: {}'.format(self.project.model.stlmodel_list[i].name))
            del self.project.model.stlmodel_list[i]
        # delete selected mathmodel
        indexes = self.ui.tableView_mathmodels.selectionModel().selectedRows()
        if len(indexes) != 0:
            i = indexes[0].row()  # 只删除选中的第一行
            self.consolePrint('Delete MATH model: {}'.format(self.project.model.mathmodel_list[i].name))
            del self.project.model.mathmodel_list[i]
        self.refreshTableViews()

    def refreshTableViews(self):
        n_stlmodels = len(self.project.model.stlmodel_list)
        n_mathmodels = len(self.project.model.mathmodel_list)

        self.tableModel_stlmodels = QStandardItemModel(n_stlmodels, 2)
        self.tableModel_stlmodels.setHorizontalHeaderLabels(['stl model', 'sld'])
        self.ui.tableView_stlmodels.setModel(self.tableModel_stlmodels)
        #self.ui.tableView_stlmodels.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)  # 横向填满
        #self.ui.tableView_stlmodels.verticalHeader().setSectionResizeMode(QHeaderView.Stretch)  # 纵向填满
        self.ui.tableView_stlmodels.setSelectionBehavior(QAbstractItemView.SelectRows)#设置只能选中整行
        for i in range(len(self.project.model.stlmodel_list)):
            stlmodel = self.project.model.stlmodel_list[i]
            item1 = QStandardItem(stlmodel.name)
            self.tableModel_stlmodels.setItem(i, 0, item1)
            item2 = QStandardItem(str(stlmodel.sld))
            self.tableModel_stlmodels.setItem(i, 1, item2)

        self.tableModel_mathmodels = QStandardItemModel(n_mathmodels, 1)
        self.tableModel_mathmodels.setHorizontalHeaderLabels(['math model'])
        self.ui.tableView_mathmodels.setModel(self.tableModel_mathmodels)
        self.ui.tableView_mathmodels.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)  # 横向填满
        #self.ui.tableView_stlmodels.verticalHeader().setSectionResizeMode(QHeaderView.Stretch)  # 纵向填满
        self.ui.tableView_mathmodels.setSelectionMode(QAbstractItemView.SingleSelection)  #设置只能选中一行
        for i in range(len(self.project.model.mathmodel_list)):
            mathmodel = self.project.model.mathmodel_list[i]
            item1 = QStandardItem(mathmodel.name)
            self.tableModel_mathmodels.setItem(i, 0, item1)
    
    def readStlmodelTableSld(self):
        for i in range(len(self.project.model.stlmodel_list)):
            sld = float(self.tableModel_stlmodels.index(i, 1).data())
            self.project.model.stlmodel_list[i].sld = sld

    def showStlModels(self):
        try:
            self.readStlmodelTableSld()
            indexes = self.ui.tableView_stlmodels.selectionModel().selectedRows()
            #self.consolePrint([index.row() for index in indexes])
            mesh_list, label_list = [], []
            for index in indexes:
                i = index.row()
                mesh_list.append(self.project.model.stlmodel_list[i].mesh)
                label_list.append(self.project.model.stlmodel_list[i].name)
            if len(mesh_list) == 0:
                mesh_list = [stlmodel.mesh for stlmodel in self.project.model.stlmodel_list]
                label_list = [stlmodel.name for stlmodel in self.project.model.stlmodel_list]
            #self.consolePrint(label_list)
            canvas = Figure_Canvas()
            plotStlMeshes(mesh_list, label_list=label_list, show=False, figure=canvas.figure)
            graphicScene = QtWidgets.QGraphicsScene()
            graphicScene.addWidget(canvas)
            stlmodelView = stlmodelViewWindow()
            stlmodelView.graphicsView.setScene(graphicScene)
            self.ui.mdiArea.addSubWindow(stlmodelView)
            stlmodelView.show()
        except:
            self.consolePrint('(X) There is no model to show...')
    def showMathModel(self):
        try:
            index = self.ui.tableView_mathmodels.currentIndex()
            i = index.row()
            canvas = Figure_Canvas()
            plotPointsWithSld(self.project.model.mathmodel_list[i].sample_points_with_sld, show=False, figure=canvas.figure)
            graphicScene = QtWidgets.QGraphicsScene()
            graphicScene.addWidget(canvas)
            mathmodelView = mathmodelViewWindow()
            mathmodelView.graphicsView.setScene(graphicScene)
            self.ui.mdiArea.addSubWindow(mathmodelView)
            mathmodelView.show()
        except:
            self.consolePrint('(X) There is no model to show...')
    def showPointsWithSld(self):
        canvas = Figure_Canvas()
        plotPointsWithSld(self.project.points_with_sld, show=False, figure=canvas.figure)
        graphicScene = QtWidgets.QGraphicsScene()
        graphicScene.addWidget(canvas)
        pointsWithSldView = pointsWithSldViewWindow()
        pointsWithSldView.graphicsView.setScene(graphicScene)
        interval = self.project.model.interval
        pointsWithSldView.label_interval.setText('interval = {:.4f}\tnumber of points = {}'.format(interval, self.project.model.points_with_sld.shape[0]))
        self.ui.mdiArea.addSubWindow(pointsWithSldView)
        pointsWithSldView.show()
    def showSasCurve(self):
        canvas = Figure_Canvas()
        plotSasCurve(self.project.data.q, self.project.data.I, show=False, figure=canvas.figure)
        graphicScene = QtWidgets.QGraphicsScene()
        graphicScene.addWidget(canvas)
        sasdataView = sasdataViewWindow()
        sasdataView.graphicsView.setScene(graphicScene)
        self.ui.mdiArea.addSubWindow(sasdataView)
        sasdataView.show()

    def genPoints(self):
        if len(self.project.model.stlmodel_list)==0 and len(self.project.model.mathmodel_list)==0:
            self.consolePrint('(X) Please import model(s) first !')
        else:
            self.readStlmodelTableSld()
            grid_num = self.ui.lineEdit_gridPointsNum.text()
            interval = self.ui.lineEdit_interval.text()
            self.consolePrint('Calculating points model...Please wait...')
            self.setPushButtonEnable(False)
            self.setProgressBarRolling(True)
            # 异步线程genPoints
            if interval != '':
                interval = float(interval)
                self.thread_genPoints = Thread_genPoints(self.project, interval=interval)
            else:
                grid_num = int(grid_num)
                self.thread_genPoints = Thread_genPoints(self.project, grid_num=grid_num)
            self.thread_genPoints.threadEnd.connect(self.processGenPointsThreadOutput)
            self.thread_genPoints.start()
    def processGenPointsThreadOutput(self, temp_project):
        self.project = temp_project
        self.consolePrint('Points model generated')
        self.showPointsWithSld()
        self.setPushButtonEnable(True)
        self.setProgressBarRolling(False)

    def calcSas(self):
        try:
            self.project.model.points_with_sld
        except:
            self.consolePrint('(X) Please generate points first !')
        else:
            self.project.setupData()
            qmin = float(self.ui.lineEdit_qmin.text())
            qmax = float(self.ui.lineEdit_qmax.text())
            qnum = int(self.ui.lineEdit_qnum.text())
            lmax = int(self.ui.lineEdit_lmax.text())
            q = self.project.data.genQ(qmin, qmax, qnum=qnum)
            self.project.data.q = q
            self.project.data.lmax = lmax
            parallel = self.ui.checkBox_parallel.isChecked()
            cpu_usage = float(self.ui.lineEdit_cpuUsage.text())
            proc_num = self.ui.lineEdit_processNum.text()
            if proc_num != '':
                proc_num = int(proc_num)
            else:
                proc_num = None

            self.setPushButtonEnable(False)
            self.setProgressBarRolling(True)
            self.consolePrint('Calculating SAS curve...Please wait...')
            # 异步线程计算SAS
            points = self.project.data.points
            sld = self.project.data.sld
            self.thread_calcSas = Thread_calcSas(q, points, sld, lmax, parallel, cpu_usage, proc_num)  # 这里的thread写成self.thread是为了防止start之后这个方法结束这个变量被回收了，会导致错误：QThread: Destroyed while thread is still running
            self.thread_calcSas.threadEnd.connect(self.processCalcSasThreadOutput)
            self.begintime = time.time()
            self.thread_calcSas.start()
    def processCalcSasThreadOutput(self, I):
        endtime = time.time()
        self.project.data.I = I
        self.project.data.error = 0.001 * I  # 默认生成千分之一的误差，主要用于写文件的占位
        self.showSasCurve()
        self.setPushButtonEnable(True)
        self.setProgressBarRolling(False)
        self.consolePrint('SAS curve calculation finished. Time consumed: {} sec'.format(round(endtime-self.begintime, 2)))

    
    ####### some functions for GUI use ########
    def consolePrint(self, string):
        console_str = '[{}] {}'.format(time.strftime('%Y-%m-%d %H:%M:%S'), string)
        print(console_str)
        self.ui.textEdit.append(console_str)
    def setPushButtonEnable(self, true_or_false):
        # 在某些计算过程中禁用一些按钮避免被疯狂点击
        self.ui.pushButton_genPoints.setEnabled(true_or_false)
        self.ui.pushButton_calcSas.setEnabled(true_or_false)
    def setProgressBarRolling(self, true_or_false):
        if true_or_false:
            # progress bar 开始滚动
            self.ui.progressBar.setMinimum(0)
            self.ui.progressBar.setMaximum(0)
        else:
            # progress bar 停止滚动
            self.ui.progressBar.setMinimum(0)
            self.ui.progressBar.setMaximum(100)
    ###########################################




if __name__ == '__main__':
    app = QApplication(sys.argv)
    Mainwindow = QMainWindow()
    ui = Ui_mainWindow()
    ui.setupUi(Mainwindow)
    func = mainwindowFunction(ui)
    Mainwindow.show()
    sys.exit(app.exec_())