# -*- coding: utf-8 -*-

from Plot import plotPointsWithSld, plotStlMeshes
from qtgui.modelModifyWindow_ui import Ui_modelModifyWindow

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QMainWindow

import matplotlib
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT
from matplotlib.figure import Figure

import numpy as np


class MplCanvas(FigureCanvasQTAgg):
    def __init__(self, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        super().__init__(self.fig)


class modelModifyWindow(QMainWindow, Ui_modelModifyWindow):

    def __init__(self, project):
        super().__init__()
        self.setupUi(self)
        self.project = project
        self.setModelComboBox()

        # add the canvas and toolbar
        self.canvas = MplCanvas()
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        frame_plot_Layout = QtWidgets.QVBoxLayout()
        frame_plot_Layout.addWidget(self.toolbar)
        frame_plot_Layout.addWidget(self.canvas)
        self.frame_plot.setLayout(frame_plot_Layout)

        self.comboBox_models.currentIndexChanged.connect(self.modelSelected)
        self.pushButton_translate.clicked.connect(self.translateModel)
        self.pushButton_rotate.clicked.connect(self.rotateModel)


    def setModelComboBox(self):
        stlmodelname_list = [model.name for model in self.project.model.stlmodel_list]
        mathmodelname_list = [model.name for model in self.project.model.mathmodel_list]
        modelname_list = stlmodelname_list + mathmodelname_list
        self.comboBox_models.addItems(modelname_list)
        self.stlmodel_num = len(stlmodelname_list)
        self.mathmodel_num = len(mathmodelname_list)
        self.comboBox_models.setCurrentIndex(-1)  # 设置默认选项为空

    def modelSelected(self, i):
        # i is the index changed, int
        if i < self.stlmodel_num:  # means that it is stlmodel selected
            index = i
            self.model = self.project.model.stlmodel_list[index]
            self.model_type = 'STL'
        else:  # means that it is mathmodel selected
            index = i - self.stlmodel_num
            self.model = self.project.model.mathmodel_list[index]
            self.model_type = 'PY'
        self.label_modelType.setText('model type: {}'.format(self.model_type))
        self.showModel()

    def showModel(self):
        #axes = self.canvas.fig.gca()
        #axes.cla()
        self.canvas.fig.clf()  # 清空了figure, 没有axes了
        self.canvas.fig.canvas.draw_idle()  # 即使清空了figure和axes, 但是画布中可能还存在残留数据，不用这个方法的话就不会即时更新图像
        #print(len(self.canvas.fig.axes))  # figure.axes 参数是figure的axes列表
        if self.model_type == 'STL':
            plotStlMeshes([self.model.mesh], label_list=[self.model.name], show=False, figure=self.canvas.fig)
        elif self.model_type == 'PY':
            plotPointsWithSld(self.model.sample_points_with_sld, show=False, figure=self.canvas.fig)
        
    def translateModel(self):
        x = float(self.lineEdit_translateX.text())
        y = float(self.lineEdit_translateY.text())
        z = float(self.lineEdit_translateZ.text())
        self.model.translate([x, y, z])
        self.showModel()

    def rotateModel(self):
        axis_x = float(self.lineEdit_axisX.text())
        axis_y = float(self.lineEdit_axisY.text())
        axis_z = float(self.lineEdit_axisZ.text())
        angle = float(self.lineEdit_angle.text()) / 180 * np.pi
        center_x = float(self.lineEdit_centerX.text())
        center_y = float(self.lineEdit_centerY.text())
        center_z = float(self.lineEdit_centerZ.text())
        axis = [axis_x, axis_y, axis_z]
        center = [center_x, center_y, center_z]
        self.model.rotate(axis, angle, point=center)
        self.showModel()


if __name__ == '__main__':
    from PyQt5.QtWidgets import QApplication
    import sys
    from Model2SAS import model2sas

    project = model2sas('test')
    project.importFile('./models/rod2.py')
    project.importFile('./models/torus.STL')
    project.importFile('./models/square_frame.py')
    app = QApplication(sys.argv)
    w = modelModifyWindow(project)
    w.show()
    app.exec_()