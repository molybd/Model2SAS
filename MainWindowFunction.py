# -*- coding: UTF-8 -*-

import os
import zipfile
import json
import time
import numpy as np

from Model2SAS import *
from Plot import *
from Functions import intensity_parallel, intensity

# 以下均为GUI相关的导入
import sys
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import  QWidget, QApplication, QMainWindow, QMdiSubWindow, QFileDialog, QDialog, QInputDialog, QHeaderView, QAbstractItemView
from PyQt5.QtGui import QStandardItemModel, QStandardItem, QImage, QPainter

# needed for plot
import matplotlib
matplotlib.use("Qt5Agg")  # 声明使用QT5
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

# my own qtgui files
from qtgui.mainwindow_ui import Ui_mainWindow
from qtgui.plotView_ui import Ui_plotView

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
(solved) 3. 保存project
程序结构：
(solved) 1. genPoints() 异步进行
(solved) 2. control panel 变成dock widget
'''

class EmittingStream(QtCore.QObject):
    '''写一个信号，用来发射标准输出作为信号，为了在console中显示print的值和错误信息
    '''
    textWritten = QtCore.pyqtSignal(str)  #定义一个发送str的信号
    def write(self, text):
        self.textWritten.emit(str(text))  

class Thread_calcSas(QThread):
    # 线程结束的signal，并且带有一个ndarray参数
    threadEnd = pyqtSignal(np.ndarray)
    def __init__(self, q, points, sld, lmax, parallel, core_num, proc_num):
        super(Thread_calcSas, self).__init__()
        self.q = q
        self.points = points
        self.sld = sld
        self.lmax = lmax
        self.parallel = parallel
        self.core_num = core_num
        self.proc_num = proc_num
    def run(self):
        # 线程所需要执行的代码
        if self.parallel:
            self.I = intensity_parallel(self.q, self.points, self.sld, self.lmax, core_num=self.core_num, proc_num=self.proc_num)
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
        self.ui.actionLoad_project.triggered.connect(self.loadProject)
        self.ui.actionSave_project.triggered.connect(self.saveProject)
        self.ui.actionSave_points_model.triggered.connect(self.savePointsModel)
        self.ui.actionSave_SAS_curve.triggered.connect(self.saveSasCurve)
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
        
        self.temp_folder = './.TEMP_Model2SAS'
        if not os.path.exists(self.temp_folder):
            os.mkdir(self.temp_folder)
        
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

    def loadProject(self):
        self.clearTempFolder()
        temp_folder = self.temp_folder
        filename, filetype = QFileDialog.getOpenFileName(None, 'Load project file', './', "zip Files (*.zip)")
        if filename:
            project_name = os.path.splitext(os.path.basename(filename))[0]
            self.project = model2sas(project_name)
            self.ui.label_projectName.setText('Project: {}'.format(self.project.name))
            with zipfile.ZipFile(filename, mode='r') as z:
                z.extractall(path=temp_folder)

            modelfile_list = []
            for file in os.listdir(temp_folder):
                filepath = os.path.join(temp_folder, file)
                extension = file.split('.')[-1].lower()
                if extension == 'stl' or extension == 'py':
                    modelfile_list.append(filepath)
                elif extension == 'txt':
                    filepath_points_with_sld = filepath
                elif extension == 'json':
                    filepath_info = filepath
                elif extension == 'dat':
                    filepath_data = filepath

            for filepath in modelfile_list:
                self.project.importFile(filepath, sld=1)

            self.project.model.points_with_sld = np.loadtxt(filepath_points_with_sld)
            self.project.points_with_sld = self.project.model.points_with_sld
            self.project.setupData()

            with open(filepath_info, 'r') as f:
                info_dict = json.load(f)
            self.project.model.interval = float(info_dict['interval'])
            self.project.data.lmax = int(info_dict['lmax'])

            data = np.loadtxt(filepath_data)
            q, I, error = data[:,0].flatten(), data[:,1].flatten(), data[:,2].flatten()
            self.project.data.q, self.project.data.I, self.project.data.error = q, I, error
            self.project.data.q, self.project.data.I = q, I

            self.ui.lineEdit_interval.setText(str(self.project.model.interval))
            self.ui.lineEdit_lmax.setText(str(self.project.data.lmax))
            self.refreshTableViews()
            self.showPointsWithSld()
            self.showSasCurve()
            self.clearTempFolder()
            self.consolePrint('Load project file {}'.format(filename))

    def saveProject(self):
        self.clearTempFolder()
        try:
            self.project.points_with_sld
            self.project.data.I
        except:
            self.consolePrint('(X) Project not completed...')
        else:
            folder = QFileDialog.getExistingDirectory(None, 'Select project saving directory', './')
            if folder:
                project_name = self.project.name
                filelist = []
                for stlmodel in self.project.model.stlmodel_list:
                    filelist.append(stlmodel.filepath)
                for mathmodel in self.project.model.mathmodel_list:
                    filelist.append(mathmodel.filepath)
                
                temp_folder = self.temp_folder
                filepath_points_model = os.path.join(temp_folder, 'PointsModel.txt')
                filepath_sas_curve = os.path.join(temp_folder, 'SasCurve.dat')
                self.project.savePointsWithSld(filepath_points_model)
                self.project.saveSasData(filepath_sas_curve)
                filelist.append(filepath_points_model)
                filelist.append(filepath_sas_curve)

                info_dict = {
                    'interval': self.project.model.interval,
                    'lmax': self.project.data.lmax
                }
                filepath_info = os.path.join(temp_folder, 'info.json')
                with open(filepath_info, 'w') as f:
                    json.dump(info_dict, f)
                filelist.append(filepath_info)

                filepath_zip = os.path.join(folder, '{}.zip'.format(project_name))
                with zipfile.ZipFile(filepath_zip, mode='w', compression=zipfile.ZIP_STORED) as z:
                    for filepath in filelist:
                        z.write(filepath, arcname=os.path.basename(filepath))
                self.consolePrint('Project saved in {}'.format(os.path.abspath(filepath_zip)))
                self.clearTempFolder()
    def savePointsModel(self):
        try:
            self.project.points_with_sld
        except:
            self.consolePrint('(X) There is no points model to save...')
        else:
            filename, filetype = QFileDialog.getSaveFileName(None, 'Save points model file', './', "txt Files (*.txt)")
            if filename:
                self.project.savePointsWithSld(filename)
                self.consolePrint('Points model saved in {}'.format(filename))
    def saveSasCurve(self):
        try:
            self.project.data.I
        except:
            self.consolePrint('(X) There is no data to save...')
        else:
            filename, filetype = QFileDialog.getSaveFileName(None, 'Save SAS data file', './', "data Files (*.dat)")
            if filename:
                self.project.saveSasData(filename)
                self.consolePrint('SAS data saved in {}'.format(filename))

    def clearTempFolder(self):
        def clearFolder(folder):
            for path in os.listdir(folder):
                path = os.path.join(folder, path)
                if os.path.isfile(path):
                    os.remove(path)
                elif os.path.isdir(path):
                    clearFolder(path)
                    os.rmdir(path)
        folder = self.temp_folder
        clearFolder(folder)
        
    
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
            
            window = Ui_plotView()
            plotStlMeshes(mesh_list, label_list=label_list, show=False, figure=window.canvas.fig)
            window.setWindowTitle('STL model view')
            window.label_text.setText('')
            self.ui.mdiArea.addSubWindow(window)
            window.show()
        except:
            self.consolePrint('(X) There is no model to show...')
    def showMathModel(self):
        try:
            index = self.ui.tableView_mathmodels.currentIndex()
            i = index.row()
            window = Ui_plotView()
            plotPointsWithSld(self.project.model.mathmodel_list[i].sample_points_with_sld, show=False, figure=window.canvas.fig)
            window.setWindowTitle('Math model view')
            window.label_text.setText('')
            self.ui.mdiArea.addSubWindow(window)
            window.show()
        except:
            self.consolePrint('(X) There is no model to show...')
    def showPointsWithSld(self):
        window = Ui_plotView()
        plotPointsWithSld(self.project.points_with_sld, show=False, figure=window.canvas.fig)
        window.setWindowTitle('Points with SLD view')
        interval = self.project.model.interval
        text = 'interval = {:.4f}\tnumber of points = {}'.format(interval, self.project.model.points_with_sld.shape[0])
        window.label_text.setText(text)
        self.ui.mdiArea.addSubWindow(window)
        window.show()
    def showSasCurve(self):
        window = Ui_plotView()
        plotSasCurve(self.project.data.q, self.project.data.I, show=False, figure=window.canvas.fig)
        window.setWindowTitle('SAS curve view')
        window.label_text.setText('')
        self.ui.mdiArea.addSubWindow(window)
        window.show()


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
            self.begintime = time.time()
            self.thread_genPoints.start()
    def processGenPointsThreadOutput(self, temp_project):
        endtime = time.time()
        self.project = temp_project
        self.showPointsWithSld()
        self.setPushButtonEnable(True)
        self.setProgressBarRolling(False)
        self.consolePrint('Points model generation finished. Time consumed: {} sec'.format(round(endtime-self.begintime, 2)))

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
            core_num_text = self.ui.lineEdit_coreNum.text()
            proc_num_text = self.ui.lineEdit_processNum.text()
            try:
                core_num = int(core_num_text)
            except:
                core_num = 2
                self.ui.lineEdit_coreNum.setText('2')
            try:
                proc_num = int(proc_num_text)
            except:
                proc_num = 4
                self.ui.lineEdit_progressNum.setText('4')

            self.setPushButtonEnable(False)
            self.setProgressBarRolling(True)
            self.consolePrint('Calculating SAS curve...See CLI for progress...')
            # 异步线程计算SAS
            points = self.project.data.points
            sld = self.project.data.sld
            self.thread_calcSas = Thread_calcSas(q, points, sld, lmax, parallel, core_num, proc_num)  # 这里的thread写成self.thread是为了防止start之后这个方法结束这个变量被回收了，会导致错误：QThread: Destroyed while thread is still running
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