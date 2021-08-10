# -*- coding: UTF-8 -*-

from qtgui.modelModifyWindow_ui import Ui_modelModifyWindow
from ModelModifyWindow import modelModifyWindow
import os
#import zipfile
#import json
import time

from Model2SAS import model2sas
from Plot import *

# 以下均为GUI相关的导入
import sys

from PyQt5 import QtWidgets, QtCore
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtCore import QThread, Qt, pyqtSignal

# my own qtgui files
from qtgui.mainwindow_ui import Ui_mainWindow
from qtgui.plotView_ui import Ui_plotView
from ModelModifyWindow import modelModifyWindow


class Thread_genLatticeModel(QThread):
    threadEnd = pyqtSignal(object)
    def __init__(self, project, grid_num=10000, interval=None):
        super().__init__()
        self.temp_project = project
        self.grid_num = grid_num
        self.interval = interval
    def run(self):
        if self.interval:
            self.temp_project.genPoints(interval=self.interval)
        elif self.grid_num:
            self.temp_project.genPoints(grid_num=self.grid_num)
        self.threadEnd.emit(self.temp_project)

class Thread_calcSasCurve(QThread):
    threadEnd = pyqtSignal(object)
    def __init__(self, project, qmin, qmax, qnum, lmax):
        super().__init__()
        self.temp_project = project
        self.qmin, self.qmax, self.qnum, self.lmax = qmin, qmax, qnum, lmax
    def run(self):
        self.temp_project.calcSas(self.qmin, self.qmax, self.qnum, lmax=self.lmax)
        self.threadEnd.emit(self.temp_project)


class TableModel(QtCore.QAbstractTableModel):
    # for display model list 
    def __init__(self, header_labels, models=None):
        super().__init__()
        self.model_list = models or []
        self.header_labels = header_labels
    def data(self, index, role):
        if role == Qt.DisplayRole:
        # .row() indexes into the outer list
            model_name = self.model_list[index.row()]
            return model_name
    def headerData(self, section, orientation, role=Qt.DisplayRole):
        # set header for the table view
        if role == Qt.DisplayRole and orientation == Qt.Horizontal:
            return self.header_labels[section]
    def rowCount(self, index):
        # The length of the list.
        # only rows in ListModel
        return len(self.model_list)
    def columnCount(self, index):
        return 1


class MainWindow(QtWidgets.QMainWindow, Ui_mainWindow):

    def __init__(self):
        super().__init__()
        self.setupUi(self)

        defalult_name = 'New Project'
        self.project = model2sas(defalult_name)
        self.label_projectName.setText('Project: {}'.format(self.project.name))

        self.tableModel_stlModels = TableModel(['STL model'])
        self.tableModel_mathModels = TableModel(['MATH model'])
        self.tableView_stlModels.setModel(self.tableModel_stlModels)
        self.tableView_mathModels.setModel(self.tableModel_mathModels)
        self.tableView_stlModels.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)  # 横向填满
        self.tableView_mathModels.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)  # 横向填满

        self.pushButton_importModels.clicked.connect(self.importModels)
        self.pushButton_deleteModels.clicked.connect(self.deleteSelectedModels)
        self.pushButton_showSelectedModels.clicked.connect(self.showSelectedModels)
        self.pushButton_genLatticeModel.clicked.connect(self.genLatticeModel)
        self.pushButton_calcSas.clicked.connect(self.calcSasCurve)
        self.pushButton_showAllModels.clicked.connect(self.showAllModels)
        self.pushButton_modifyModel.clicked.connect(self.showModelModifyWindow)

        self.action_deleteAllModels.triggered.connect(self.deleteAllModels)

    def importModels(self):
        filepath_list, filetype_list = QFileDialog.getOpenFileNames(None, 'Select Model File(s)', './', "All Files (*);;stl Files (*.stl);;math model Files (*.py)")
        for filepath in filepath_list:
            self.project.importFile(filepath, sld=1)
            modelname, filetype = os.path.splitext(os.path.basename(filepath))
            filetype = filepath.split('.')[-1].upper()
            self.consolePrint(
                'Import {} model with path: {}'.format(filetype, filepath)
            )
            if filetype == 'STL':
                self.tableModel_stlModels.model_list.append(modelname)
            elif filetype == 'PY':
                self.tableModel_mathModels.model_list.append(modelname)
        self.tableModel_stlModels.layoutChanged.emit()
        self.tableModel_mathModels.layoutChanged.emit()

    def deleteSelectedModels(self):
        # read the selected model indexes
        indexes_stlmodel = [selectedRow.row() for selectedRow in self.tableView_stlModels.selectionModel().selectedRows()]
        indexes_mathmodel = [selectedRow.row() for selectedRow in self.tableView_mathModels.selectionModel().selectedRows()]
        indexes_stlmodel.sort(reverse=True)
        indexes_mathmodel.sort(reverse=True)
        for i in indexes_stlmodel:
            self.consolePrint('Delete STL model: {}'.format(self.project.model.stlmodel_list[i].name))
            del self.tableModel_stlModels.model_list[i]
            del self.project.model.stlmodel_list[i]
        for i in indexes_mathmodel:
            self.consolePrint('Delete MATH model: {}'.format(self.project.model.mathmodel_list[i].name))
            del self.tableModel_mathModels.model_list[i]
            del self.project.model.mathmodel_list[i]
        self.tableModel_stlModels.layoutChanged.emit()
        self.tableModel_mathModels.layoutChanged.emit()

    def showSelectedModels(self):
        # read the selected model indexes
        indexes_stlmodel = [selectedRow.row() for selectedRow in self.tableView_stlModels.selectionModel().selectedRows()]
        indexes_mathmodel = [selectedRow.row() for selectedRow in self.tableView_mathModels.selectionModel().selectedRows()]
        indexes_stlmodel.sort()
        indexes_mathmodel.sort()

        mesh_list = [self.project.model.stlmodel_list[i].mesh for i in indexes_stlmodel]
        label_list = [self.project.model.stlmodel_list[i].name for i in indexes_stlmodel]
        if len(mesh_list) != 0:
            window_stlmodel = Ui_plotView()
            plotStlMeshes(mesh_list, label_list=label_list, show=False, figure=window_stlmodel.canvas.fig)
            window_stlmodel.setWindowTitle('STL model view')
            window_stlmodel.label_text.setText('')
            self.mdiArea.addSubWindow(window_stlmodel)
            window_stlmodel.show()

        points_list = [self.project.model.mathmodel_list[i].sample_points for i in indexes_mathmodel]
        label_list = [self.project.model.mathmodel_list[i].name for i in indexes_mathmodel]
        if len(points_list) != 0:
            window_mathmodel = Ui_plotView()
            plotMultiplePoints(points_list, label_list, show=False, figure=window_mathmodel.canvas.fig)
            window_mathmodel.setWindowTitle('MATH model view')
            window_mathmodel.label_text.setText('')
            self.mdiArea.addSubWindow(window_mathmodel)
            window_mathmodel.show()
        if len(indexes_stlmodel) == 0 and len(indexes_mathmodel) == 0:
            self.consolePrint('(!) No model selected.')
        else:
            self.consolePrint('Selected models shown.')

    def showAllModels(self):
        mesh_list = [model.mesh for model in self.project.model.stlmodel_list]
        label_list = [model.name for model in self.project.model.stlmodel_list]
        if len(mesh_list) != 0:
            window_stlmodel = Ui_plotView()
            plotStlMeshes(mesh_list, label_list=label_list, show=False, figure=window_stlmodel.canvas.fig)
            window_stlmodel.setWindowTitle('STL model view')
            window_stlmodel.label_text.setText('')
            self.mdiArea.addSubWindow(window_stlmodel)
            window_stlmodel.show()

        points_list = [model.sample_points for model in self.project.model.mathmodel_list]
        label_list = [model.name for model in self.project.model.mathmodel_list]
        if len(points_list) != 0:
            window_mathmodel = Ui_plotView()
            plotMultiplePoints(points_list, label_list, show=False, figure=window_mathmodel.canvas.fig)
            window_mathmodel.setWindowTitle('MATH model view')
            window_mathmodel.label_text.setText('')
            self.mdiArea.addSubWindow(window_mathmodel)
            window_mathmodel.show()

        if len(mesh_list) == 0 and len(points_list) == 0:
            self.consolePrint('(!) Please import model first.')
        else:
            self.consolePrint('All models shown.')

    def deleteAllModels(self):
        self.project.model.stlmodel_list = []
        self.project.model.mathmodel_list = []
        self.tableModel_stlModels.model_list = []
        self.tableModel_mathModels.model_list = []
        self.tableModel_stlModels.layoutChanged.emit()
        self.tableModel_mathModels.layoutChanged.emit()
        self.consolePrint('All models deleted.')

    def showModelModifyWindow(self):
        window_model_modify = modelModifyWindow(self.project)
        window_model_modify.setWindowTitle('Modify Model')
        self.mdiArea.addSubWindow(window_model_modify)
        window_model_modify.show()

    def genLatticeModel(self):
        if len(self.project.model.stlmodel_list)==0 and len(self.project.model.mathmodel_list)==0:
            self.consolePrint('(!) Please import model first !')
        else:
            grid_num = self.lineEdit_gridPointsNum.text()
            interval = self.lineEdit_interval.text()
            self.consolePrint('Calculating lattice model...Please wait...')
            self.setPushButtonEnable(False)
            self.setProgressBarRolling(True)
            # 异步计算
            if interval != '':
                interval = float(interval)
                self.thread = Thread_genLatticeModel(self.project, interval=interval)
            else:
                grid_num = int(grid_num)
                self.thread = Thread_genLatticeModel(self.project, grid_num=grid_num)
            self.thread.threadEnd.connect(self.threadOutput_genLatticeModel)
            self.begintime = time.time()
            self.thread.start()
    def threadOutput_genLatticeModel(self, temp_project):
        endtime = time.time()
        self.project = temp_project
        self.setPushButtonEnable(True)
        self.setProgressBarRolling(False)
        self.consolePrint('Points model generation finished. Time consumed: {:.2f} sec'.format(endtime-self.begintime))
        self.showLatticeModel()

    def showLatticeModel(self):
        window = Ui_plotView()
        plotPointsWithSld(self.project.points_with_sld, show=False, figure=window.canvas.fig)
        window.setWindowTitle('Lattice model view')
        interval = self.project.model.interval
        text = 'interval = {:.4f}\tnumber of points = {}'.format(interval, self.project.model.points_with_sld.shape[0])
        window.label_text.setText(text)
        self.mdiArea.addSubWindow(window)
        window.show()

    def calcSasCurve(self):
        try:
            self.project.model.points_with_sld
        except:
            self.consolePrint('(X) Please generate points first !')
        else:
            self.project.setupData()
            qmin = float(self.lineEdit_qmin.text())
            qmax = float(self.lineEdit_qmax.text())
            qnum = int(self.lineEdit_qnum.text())
            lmax = int(self.lineEdit_lmax.text())
            self.setPushButtonEnable(False)
            self.setProgressBarRolling(True)
            self.consolePrint('Calculating SAS curve...See CLI for progress...')
            #异步计算
            self.thread = Thread_calcSasCurve(self.project, qmin, qmax, qnum, lmax)
            self.thread.threadEnd.connect(self.threadOutput_calcSasCurve)
            self.begintime = time.time()
            self.thread.start()
    def threadOutput_calcSasCurve(self, temp_project):
        endtime = time.time()
        self.project = temp_project
        self.setPushButtonEnable(True)
        self.setProgressBarRolling(False)
        self.consolePrint('SAS curve calculation finished. Time consumed: {:.2f} sec'.format(endtime-self.begintime))
        self.showSasCurve()

    def showSasCurve(self):
        window = Ui_plotView()
        plotSasCurve(self.project.q, self.project.I, show=False, figure=window.canvas.fig)
        window.setWindowTitle('SAS curve view')
        window.label_text.setText('')
        self.mdiArea.addSubWindow(window)
        window.show()


    ####### some functions for GUI use ########
    def consolePrint(self, string):
        console_str = '[{}] {}'.format(time.strftime('%Y-%m-%d %H:%M:%S'), string)
        print(console_str)
        self.textEdit_console.append(console_str)
    
    def setPushButtonEnable(self, true_or_false):
        # 在某些计算过程中禁用一些按钮避免被疯狂点击
        self.pushButton_genLatticeModel.setEnabled(true_or_false)
        self.pushButton_calcSas.setEnabled(true_or_false)

    def setProgressBarRolling(self, true_or_false):
        if true_or_false:
            # progress bar 开始滚动
            self.progressBar.setMinimum(0)
            self.progressBar.setMaximum(0)
        else:
            # progress bar 停止滚动
            self.progressBar.setMinimum(0)
            self.progressBar.setMaximum(100)
    ###########################################


app = QtWidgets.QApplication(sys.argv)
window = MainWindow()
window.show()
app.exec()
