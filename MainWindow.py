# -*- coding: UTF-8 -*-

from numpy import pi
from ModelModifyWindow import modelModifyWindow
from MathModelGenerationWindow import mathModelGenerationWindow
import os
import zipfile
import json
import pickle
import time
import shutil

from Model2SAS import model2sas
from Plot import *

# 以下均为GUI相关的导入
import sys

from PyQt5 import QtWidgets, QtCore
from PyQt5.QtWidgets import QFileDialog, QMessageBox, QInputDialog
from PyQt5.QtCore import QThread, Qt, pyqtSignal
from PyQt5.QtGui import QIcon

# my own qtgui files
from qtgui.MainWindow_ui import Ui_mainWindow
from qtgui.PlotView_ui import Ui_plotView
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
    def __init__(self, project, qmin, qmax, qnum, lmax, useGpu):
        super().__init__()
        self.temp_project = project
        self.qmin, self.qmax, self.qnum, self.lmax, self.useGpu = qmin, qmax, qnum, lmax, useGpu
    def run(self):
        self.temp_project.calcSas(self.qmin, self.qmax, self.qnum, lmax=self.lmax, use_gpu=self.useGpu)
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
        self.setWindowIcon(QIcon('resource/icon/logo_256.ico'))  # set window icon

        self.temp_folder = os.path.normpath(os.path.expanduser('~/.TEMP_Model2SAS/'))  # 在用户目录下建立临时文件夹，以免安装后出现权限不足报错 set temp folder in user path
        if not os.path.exists(self.temp_folder):
            os.mkdir(self.temp_folder)
        self.clearTempFolder()

        self.default_params_json = './default_params.json'
        self.setParamsFromJson(self.default_params_json)

        self.project = model2sas(self.params['project name'])
        self.consolePrint('New project established with name: {}'.format(self.project.name))

        self.tableModel_models = TableModel(['Models'])
        self.tableView_models.setModel(self.tableModel_models)
        self.tableView_models.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)  # 横向填满 fill horizontally
        
        self.checkBox_useGpu.setEnabled(False)  # disabled before configuring gpu
        self.radioButton_gridPointsNum.setChecked(True)  # use grid num by default

        self.pushButton_importModels.clicked.connect(self.importModels)
        self.pushButton_deleteModels.clicked.connect(self.deleteSelectedModels)
        self.pushButton_showSelectedModels.clicked.connect(self.showSelectedModels)
        self.pushButton_genLatticeModel.clicked.connect(self.genLatticeModel)
        self.pushButton_calcSas.clicked.connect(self.calcSasCurve)
        self.pushButton_showAllModels.clicked.connect(self.showAllModels)
        self.pushButton_modifyModel.clicked.connect(self.showModelModifyWindow)

        self.action_newProject.triggered.connect(self.newProject)
        self.action_loadProject.triggered.connect(self.loadProject)
        self.action_saveProject.triggered.connect(self.saveProject)
        self.action_exportTxtFile.triggered.connect(self.exportLatticeModelAsTxtFile)
        self.action_exportPDBFile.triggered.connect(self.exportLatticeModelAsPDBFile)
        self.action_saveSasCurve.triggered.connect(self.saveSasCurve)
        self.action_importModels.triggered.connect(self.importModels)
        self.action_deleteAllModels.triggered.connect(self.deleteAllModels)
        self.action_configureGpu.triggered.connect(self.configureGpu)
        self.action_showLatticeModel.triggered.connect(self.showLatticeModel)
        self.action_showSasCurve.triggered.connect(self.showSasCurve)
        self.action_showAllModels.triggered.connect(self.showAllModels)
        self.action_showSelectedModels.triggered.connect(self.showSelectedModels)
        self.action_mathModelGeneration.triggered.connect(self.showMathModelGenerationWindow)
        self.action_Cascade.triggered.connect(self.mdiArea.cascadeSubWindows)
        self.action_Tile.triggered.connect(self.mdiArea.tileSubWindows)


    def importModels(self):
        filepath_list, filetype_list = QFileDialog.getOpenFileNames(None, 'Select Model File(s)', './', "All Files (*);;stl Files (*.stl);;math model Files (*.py)")
        for filepath in filepath_list:
            basename = os.path.basename(filepath)
            filetype = filepath.split('.')[-1].upper()
            if filetype != 'STL' and filetype != 'PY':
                self.consolePrint('(X) Wrong file type! Only .stl and .py file supported')
            else:
                try:
                    self.project.importFile(filepath, sld=1)
                except:
                    self.consolePrint('(X) Import model failed with path: {}'.format(filepath))
                else:
                    self.consolePrint('Import {} model with path: {}'.format(filetype, filepath))
                    shutil.copyfile(filepath, os.path.join(self.temp_folder, basename))  # 在temp_folder中存储副本，供保存project用
        self.updateTableView()

    def deleteSelectedModels(self):
        # read the selected model indexes
        indexes, indexes_stlmodel, indexes_mathmodel = self.getSelectedModelIndexes()
        indexes_stlmodel.sort(reverse=True)
        indexes_mathmodel.sort(reverse=True)
        for i in indexes_stlmodel:
            self.consolePrint('Delete STL model: {}'.format(self.project.model.stlmodel_list[i].name))
            del self.project.model.stlmodel_list[i]
        for i in indexes_mathmodel:
            self.consolePrint('Delete MATH model: {}'.format(self.project.model.mathmodel_list[i].name))
            del self.project.model.mathmodel_list[i]
        self.updateTableView()

    def showSelectedModels(self):
        # read the selected model indexes
        indexes, indexes_stlmodel, indexes_mathmodel = self.getSelectedModelIndexes()
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
        self.updateTableView()
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
            if self.radioButton_interval.isChecked():
                interval = float(interval)
                self.thread = Thread_genLatticeModel(self.project, interval=interval)
            else:
                grid_num = int(grid_num)
                self.thread = Thread_genLatticeModel(self.project, grid_num=grid_num)
            # 异步计算 asynchronization
            self.thread.threadEnd.connect(self.threadOutput_genLatticeModel)
            self.begintime = time.time()
            self.thread.start()
    def threadOutput_genLatticeModel(self, temp_project):
        endtime = time.time()
        self.project = temp_project
        self.setPushButtonEnable(True)
        self.setProgressBarRolling(False)
        self.consolePrint('Lattice model generation finished. Time consumed: {:.2f} sec'.format(endtime-self.begintime))
        self.showLatticeModel()

    def showLatticeModel(self):
        try:
            window = Ui_plotView()
            plotPointsWithSld(self.project.points_with_sld, show=False, figure=window.canvas.fig)
            window.setWindowTitle('Lattice model view')
            interval = self.project.model.interval
            text = 'interval = {:.4f}\tnumber of points = {}'.format(interval, self.project.model.points_with_sld.shape[0])
            window.label_text.setText(text)
            self.mdiArea.addSubWindow(window)
            window.show()
            self.consolePrint('Lattice model shown')
        except:
            self.consolePrint('(X) Lattice model hasn\'t been generated !')

    def calcSasCurve(self):
        try:
            self.project.model.points_with_sld
        except:
            self.consolePrint('(X) Please generate lattice model first !')
        else:
            self.project.setupData()
            qmin = float(self.lineEdit_qmin.text())
            qmax = float(self.lineEdit_qmax.text())
            qnum = int(self.lineEdit_qnum.text())
            lmax = int(self.lineEdit_lmax.text())
            useGpu = self.checkBox_useGpu.isChecked()
            self.setPushButtonEnable(False)
            self.setProgressBarRolling(True)
            if useGpu:
                self.consolePrint('Calculating SAS curve using GPU...See CLI for progress...')
            else:
                self.consolePrint('Calculating SAS curve using CPU...See CLI for progress...')
            #异步计算 asynchronization
            self.thread = Thread_calcSasCurve(self.project, qmin, qmax, qnum, lmax, useGpu)
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

    def showMathModelGenerationWindow(self):
        window_gen_math_model = mathModelGenerationWindow()
        self.mdiArea.addSubWindow(window_gen_math_model)
        window_gen_math_model.show()

    def showSasCurve(self):
        try:
            window = Ui_plotView()
            plotSasCurve(self.project.q, self.project.I, show=False, figure=window.canvas.fig)
            window.setWindowTitle('SAS curve view')
            window.label_text.setText('')
            self.mdiArea.addSubWindow(window)
            window.show()
            self.consolePrint('SAS curve shown')
        except:
            self.consolePrint('(X) SAS curve hasn\'t been generated !')

    def configureGpu(self):
        try:
            import torch
            button = QMessageBox.information(
                self, 
                'GPU configuration', 
                'Successful !\nPyTorch version: {}'.format(torch.__version__)
                )
            self.checkBox_useGpu.setEnabled(True)
        except:
            button = QMessageBox.critical(
                self, 
                '',
                'GPU configuration failed')

    def exportLatticeModelAsTxtFile(self):
        try:
            self.project.points_with_sld
        except:
            self.consolePrint('(X) There is no points model to save...')
        else:
            filename, filetype = QFileDialog.getSaveFileName(None, 'Save lattice model file', './', "txt Files (*.txt)")
            if filename:
                self.project.savePointsWithSld(filename)
                self.consolePrint('Lattice model saved in {}'.format(filename))

    def exportLatticeModelAsPDBFile(self):
        try:
            self.project.points_with_sld
        except:
            self.consolePrint('(X) There is no points model to save...')
        else:
            QtWidgets.QMessageBox.warning(self, 'Warning', 'All SLD information will be lost in PDB file !')
            filename, filetype = QFileDialog.getSaveFileName(None, 'Save lattice model file', './', "PDB Files (*.pdb)")
            if filename:
                self.project.model.exportPDBFile(filename)
                self.consolePrint('Lattice model saved in {}'.format(filename))
        

    
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

    def saveProject(self):
        filepath_save, ok = QFileDialog.getSaveFileName(
            self,
            'Save project',
            './',
            'All Files (*);;Model2SAS project Files (*.m2sproj)'
            )
        if ok:
            self.params['project name'] = self.project.name
            self.params['grid points num'] = int(self.lineEdit_gridPointsNum.text())
            self.params['interval'] = float(self.lineEdit_interval.text())
            self.params['use grid points num'] = self.radioButton_gridPointsNum.isChecked()
            self.params['use interval'] = self.radioButton_interval.isChecked()
            self.params['q min'] = float(self.lineEdit_qmin.text())
            self.params['q max'] = float(self.lineEdit_qmax.text())
            self.params['q num'] = int(self.lineEdit_qnum.text())
            self.params['l max'] = int(self.lineEdit_lmax.text())
            l = [model.name for model in self.project.model.stlmodel_list]
            l += [model.name for model in self.project.model.mathmodel_list]
            self.params['model list'] = l

            filepath_params = os.path.join(self.temp_folder, 'params.json')
            with open(filepath_params, 'w') as f:
                json.dump(self.params, f, indent=4)
            filepath_pickle = os.path.join(self.temp_folder, 'project.pickle')
            with open(filepath_pickle, 'wb') as f:
                pickle.dump(self.project, f)

            file_list = [filepath_params, filepath_pickle]
            file_list += [os.path.join(self.temp_folder, model.name) for model in self.project.model.stlmodel_list]
            file_list += [os.path.join(self.temp_folder, model.name) for model in self.project.model.mathmodel_list]
            with zipfile.ZipFile(filepath_save, mode='w', compression=zipfile.ZIP_STORED) as z:
                for file in file_list:
                    z.write(file, arcname=os.path.basename(file))
            self.consolePrint('Project saved in {}'.format(os.path.abspath(filepath_save)))

    def loadProject(self):
        filename, filetype = QFileDialog.getOpenFileName(
            self, 
            'Load project file', 
            './', 
            'Model2SAS project Files (*.m2sproj)'
            )
        if filename:
            self.clearTempFolder()
            sys.path.append(self.temp_folder)  # 这样加载数学模型时就不会出问题了 avoid errors in loading math models
            with zipfile.ZipFile(filename, mode='r') as z:
                z.extractall(path=self.temp_folder)
            for file in os.listdir(self.temp_folder):
                if file.split('.')[-1].lower() == 'json':
                    filepath_params = os.path.join(self.temp_folder, file)
                elif file.split('.')[-1].lower() == 'pickle':
                    filepath_pickle = os.path.join(self.temp_folder, file)
            with open(filepath_pickle, 'rb') as f:
                self.project = pickle.load(f)
            self.setParamsFromJson(filepath_params)
            self.updateTableView()
            self.consolePrint('Loaded project: {}'.format(filename))
            self.showAllModels()
            self.showLatticeModel()
            self.showSasCurve()

    def newProject(self):
        name, ok_pressed = QInputDialog.getText(self, 'New Project', 'Name: ')
        if ok_pressed:
            if name == '':
                name = 'New Project'
            self.project = model2sas(name)
            self.label_projectName.setText('Project: {}'.format(self.project.name))
            self.consolePrint('New project established with name: {}'.format(self.project.name))
            self.updateTableView()

    ############## some tool functions ##############
    def setParamsFromJson(self, json_file):
        try:
            with open(json_file, 'r') as f:
                self.params = json.load(f)
        except:
            self.params = {
            'project name': 'New Project',
            'model list': [],
            'grid points num': 10000,
            'interval': 1.0,
            'use grid points num': True,
            'use interval': False,
            'q min': 0.001,
            'q max': 1,
            'q num': 200,
            'l max': 50
            }  # 记录所有参数，供保存和加载project用 record all parameters for saving and loading project
            with open(json_file, 'w') as f:
                json.dump(self.params, f, indent=4)
        self.label_projectName.setText('Project: {}'.format(self.params['project name']))
        self.lineEdit_gridPointsNum.setText(str(self.params['grid points num']))
        self.lineEdit_interval.setText(str(self.params['interval']))
        if self.params['use interval']:
            self.radioButton_interval.setChecked(self.params['use interval'])
        else:
            self.radioButton_gridPointsNum.setChecked(self.params['use grid points num'])
        self.lineEdit_qmin.setText(str(self.params['q min']))
        self.lineEdit_qmax.setText(str(self.params['q max']))
        self.lineEdit_qnum.setText(str(self.params['q num']))
        self.lineEdit_lmax.setText(str(self.params['l max']))

    def clearTempFolder(self):
        def clearFolder(folder):
            for path in os.listdir(folder):
                path = os.path.join(folder, path)
                if os.path.isfile(path):
                    os.remove(path)
                elif os.path.isdir(path):
                    clearFolder(path)
                    os.rmdir(path)
        clearFolder(self.temp_folder)

    def getSelectedModelIndexes(self):
        indexes_tableview = [selectedRow.row() for selectedRow in self.tableView_models.selectionModel().selectedRows()]
        indexes_tableview.sort()
        stlmodel_num = len(self.project.model.stlmodel_list)
        indexes_stlmodel, indexes_mathmodel = [], []
        for i in indexes_tableview:
            if i < stlmodel_num:
                indexes_stlmodel.append(i)
            else:
                indexes_mathmodel.append(i-stlmodel_num)
        return indexes_tableview, indexes_stlmodel, indexes_mathmodel

    def updateTableView(self):
        self.tableModel_models.model_list = [model.name for model in self.project.model.stlmodel_list]
        self.tableModel_models.model_list += [model.name for model in self.project.model.mathmodel_list]
        self.tableModel_models.layoutChanged.emit()

    def consolePrint(self, string):
        console_str = '[{}] {}'.format(time.strftime('%Y-%m-%d %H:%M:%S'), string)
        print(console_str)
        self.textEdit_console.append(console_str)
    
    def setPushButtonEnable(self, true_or_false):
        # disable some button in case of pushed in the calculation progress
        self.pushButton_genLatticeModel.setEnabled(true_or_false)
        self.pushButton_calcSas.setEnabled(true_or_false)

    def setProgressBarRolling(self, true_or_false):
        if true_or_false:
            # progress bar begin rolling
            self.progressBar.setMinimum(0)
            self.progressBar.setMaximum(0)
        else:
            # progress bar end rolling
            self.progressBar.setMinimum(0)
            self.progressBar.setMaximum(100)
    ####################################################


app = QtWidgets.QApplication(sys.argv)
window = MainWindow()
window.show()
app.exec()
