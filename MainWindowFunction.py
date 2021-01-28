# -*- coding: UTF-8 -*-

import os
import numpy as np
from stl import mesh
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

from Model2SAS import *
from Plot import *

# 一下均为GUI相关的导入
import sys
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import  QWidget, QApplication, QMainWindow, QMdiSubWindow, QFileDialog, QDialog, QHeaderView, QAbstractItemView
from PyQt5.QtGui import QStandardItemModel, QStandardItem

# needed for plot
import matplotlib
matplotlib.use("Qt5Agg")  # 声明使用QT5
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# my own qtgui files
from qtgui.mainwindow_ui import Ui_mainWindow
from qtgui.controlPanel_ui import Ui_controlPanel
from qtgui.stlmodelView_ui import Ui_stlmodelView
from qtgui.mathmodelView_ui import Ui_mathmodelView
from qtgui.sasdataView_ui import Ui_sasdataView
from qtgui.newProject_ui import Ui_newProject


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
    def __init__(self, parent=None, figsize=(4,3), dpi=100):
        self.figure = Figure(figsize=figsize, dpi=dpi)  # 创建一个Figure，注意：该Figure为matplotlib下的figure，不是matplotlib.pyplot下面的figure
        FigureCanvas.__init__(self, self.figure) # 初始化父类
        self.setParent(parent)

class mainwindowFunction:
    
    def __init__(self, ui):
        self.ui = ui
        self.ui.actionNew_Project.triggered.connect(self.newProject)
        self.ui.actionImport_model_s.triggered.connect(self.importModels)
        self.ui.pushButton_importModels.clicked.connect(self.importModels)

        self.tableModel_stlmodels = QStandardItemModel(1, 2)
        self.tableModel_stlmodels.setHorizontalHeaderLabels(['model', 'sld'])
        self.ui.tableView_stlmodels.setModel(self.tableModel_stlmodels)
        #self.ui.tableView_stlmodels.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)  # 横向填满
        #self.ui.tableView_stlmodels.verticalHeader().setSectionResizeMode(QHeaderView.Stretch)  # 纵向填满
        self.ui.tableView_stlmodels.setSelectionBehavior(QAbstractItemView.SelectRows)#设置只能选中整行
        self.ui.pushButton_showStlmodels.clicked.connect(self.showStlModels)

        self.tableModel_mathmodels = QStandardItemModel(1, 1)
        self.tableModel_mathmodels.setHorizontalHeaderLabels(['model'])
        self.ui.tableView_mathmodels.setModel(self.tableModel_mathmodels)
        self.ui.tableView_mathmodels.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)  # 横向填满
        #self.ui.tableView_stlmodels.verticalHeader().setSectionResizeMode(QHeaderView.Stretch)  # 纵向填满
        self.ui.tableView_mathmodels.setSelectionMode(QAbstractItemView.SingleSelection)  #设置只能选中一行
        self.ui.pushButton_showMathmodel.clicked.connect(self.showMathModel)

        self.showControlPanel()
        self.ui.actionControl_Panel.triggered.connect(self.showControlPanel)


    def showControlPanel(self):
        widget = QWidget()
        controlPanel = Ui_controlPanel()
        controlPanel.setupUi(widget)
        self.ui.mdiArea.addSubWindow(widget)
        widget.show()

    def browseFolder(self):
        folder = QFileDialog.getExistingDirectory(None, 'Select Folder', './')
        self.newProjectWindow.lineEdit_path.setText(folder)
    def readNewProjectInfo(self):
        newProjectWindow = self.newProjectWindow
        name = newProjectWindow.lineEdit_name.text()
        folder = newProjectWindow.lineEdit_path.text()
        print(name, folder)
        project = model2sas(name, folder)
        project.setupModel()
        self.project = project
        self.ui.label_projectName.setText(self.project.name)
    def newProject(self):
        # new window for new project info
        dialog = QDialog()
        self.newProjectWindow = Ui_newProject()
        self.newProjectWindow.setupUi(dialog)
        self.newProjectWindow.pushButton_browse.clicked.connect(self.browseFolder)
        self.newProjectWindow.pushButton_newProject.clicked.connect(self.readNewProjectInfo)
        dialog.exec()


    def importModels(self):
        filepath_list, filetype_list = QFileDialog.getOpenFileNames(None, 'Select Model File(s)', './', "All Files (*);;stl Files (*.stl);;math model Files (*.py)")
        ###### TEST ######
        '''
        filepath_list = ['models\shell_12hole.STL', 'models\\torus.STL', 'models\\new_hollow_sphere_model.py']
        project = model2sas('test', 'models/projects')
        project.setupModel()
        self.project = project
        self.ui.label_projectName.setText(self.project.name)
        ##################
        '''
        for filepath in filepath_list:
            self.project.importFile(filepath, sld=1)
        for stlmodel in self.project.model.stlmodel_list:
            rawCount = self.tableModel_stlmodels.rowCount()
            item1 = QStandardItem(stlmodel.name)
            self.tableModel_stlmodels.setItem(rawCount-1, 0, item1)
            item2 = QStandardItem(str(stlmodel.sld))
            self.tableModel_stlmodels.setItem(rawCount-1, 1, item2)
            self.tableModel_stlmodels.insertRow(rawCount)
        for mathmodel in self.project.model.mathmodel_list:
            rawCount = self.tableModel_mathmodels.rowCount()
            item1 = QStandardItem(mathmodel.name)
            self.tableModel_mathmodels.setItem(rawCount-1, 0, item1)
            self.tableModel_mathmodels.insertRow(rawCount)

    # 仍然不显示legend！！！
    def showStlModels(self):
        indexes = self.ui.tableView_stlmodels.selectionModel().selectedRows()
        #print([index.row() for index in indexes])
        mesh_list, label_list = [], []
        for index in indexes:
            i = index.row()
            mesh_list.append(self.project.model.stlmodel_list[i].mesh)
            label_list.append(self.project.model.stlmodel_list[i].name)
        #print(label_list)
        canvas = Figure_Canvas(figsize=(5,4))
        plotStlMeshes(mesh_list, label_list=label_list, show=False, figure=canvas.figure)
        graphicScene = QtWidgets.QGraphicsScene()
        graphicScene.addWidget(canvas)
        widget = QWidget()
        stlmodelView = Ui_stlmodelView()
        stlmodelView.setupUi(widget)
        stlmodelView.graphicsView.setScene(graphicScene)
        self.ui.mdiArea.addSubWindow(widget)
        widget.show()
    def showMathModel(self):
        index = self.ui.tableView_mathmodels.currentIndex()
        print(index.row())
        i = index.row()
        canvas = Figure_Canvas(figsize=(5,4))
        plotPointsWithSld(self.project.model.mathmodel_list[i].sample_points_with_sld, show=False, figure=canvas.figure)
        graphicScene = QtWidgets.QGraphicsScene()
        graphicScene.addWidget(canvas)
        widget = QWidget()
        mathmodelView = Ui_mathmodelView()
        mathmodelView.setupUi(widget)
        mathmodelView.graphicsView.setScene(graphicScene)
        self.ui.mdiArea.addSubWindow(widget)
        widget.show()
        





if __name__ == '__main__':
    app = QApplication(sys.argv)
    Mainwindow = QMainWindow()
    ui = Ui_mainWindow()
    ui.setupUi(Mainwindow)
    func = mainwindowFunction(ui)
    Mainwindow.show()
    sys.exit(app.exec_())