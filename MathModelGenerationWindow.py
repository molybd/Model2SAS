# -*- coding: UTF-8 -*-

import os
from qtgui.MathModelGenerationWindow_ui import Ui_mathModelGenerationWindow
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QMainWindow, QFileDialog
from PyQt5.QtCore import Qt


class TableModel(QtCore.QAbstractTableModel):
    # for display mathmodel params 
    def __init__(self, data=None):
        super().__init__()
        self._data = data or []
        self.header = ['Parameter', 'Value']
    def data(self, index, role=Qt.DisplayRole):
        if index.isValid():
            if role == Qt.DisplayRole or role == Qt.EditRole:
                value = self._data[index.row()][index.column()]
                return str(value)
    def headerData(self, section, orientation, role=Qt.DisplayRole):
        # set header for the table view
        if role == Qt.DisplayRole and orientation == Qt.Horizontal:
            return self.header[section]
    def rowCount(self, index):
        # The length of the list.
        # only rows in ListModel
        return len(self._data)
    def columnCount(self, index):
        return len(self._data[0])
    def flags(self, index):  
        # 使TableView中数据可编辑必须实现的接口方法
        return Qt.ItemFlags(
                QtCore.QAbstractTableModel.flags(self, index)|
                Qt.ItemIsEnabled | Qt.ItemIsEditable | Qt.ItemIsSelectable
                )
    def setData(self, index, value, role):  
        # 连接Qt显示与背后的数据结构的方法，使数据改变和显示的改变，以及编辑的数据能够双向及时更新
        if role == Qt.EditRole:
            self._data[index.row()][index.column()] = value
            return True


class mathModelGenerationWindow(QMainWindow, Ui_mathModelGenerationWindow):

    def __init__(self):
        super().__init__()
        self.setupUi(self)

        self.lineEdit_modelName.setText('new_math_model')
        self.coord_list = ['xyz', 'sph', 'cyl']
        self.comboBox_coord.addItems(self.coord_list)
        self.comboBox_coord.setCurrentIndex(0)  # 设置默认选项 'xyz'
        self.statement_grid2xyz = 'x, y, z = grid_in_coord[:, 0], grid_in_coord[:, 1], grid_in_coord[:, 2]'
        self.coordSelected(0)
        self.tableModel_params = TableModel(data=[['', '']])
        self.tableView_params.setModel(self.tableModel_params)
        self.tableView_params.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)

        self.comboBox_coord.currentIndexChanged.connect(self.coordSelected)
        self.pushButton_addParam.clicked.connect(self.addParamLine)
        self.pushButton_saveModel.clicked.connect(self.saveMathModel)
        self.pushButton_deleteParam.clicked.connect(self.deleteSelectedParamLine)

    def addParamLine(self):
        self.tableModel_params._data.append(['', ''])
        self.tableModel_params.layoutChanged.emit()

    def deleteSelectedParamLine(self):
        indexes_tableview = [index.row() for index in self.tableView_params.selectedIndexes()]
        indexes_tableview.sort(reverse=True)
        for i in indexes_tableview:
            del self.tableModel_params._data[i]
        self.tableModel_params.layoutChanged.emit()

    def coordSelected(self, i):
        self.coord = self.coord_list[i]
        if self.coord == 'xyz':
            self.statement_variable_in_coord = 'x, y, z'
        elif self.coord == 'sph':
            self.statement_variable_in_coord = 'r, theta, phi'
        elif self.coord == 'cyl':
            self.statement_variable_in_coord = 'rho, theta, z'
        self.label_grid2xyz.setText('{} = grid_points'.format(self.statement_variable_in_coord))
        
    def genMathModelString(self):
        temp_list = self.tableModel_params._data

        # 防止有空的未填的参数表格项
        params_list = []
        for l in temp_list:
            if not '' in l:
                params_list.append(l)
        
        # generate params dict statement
        s_list = []
        for name, value in params_list:
            s_list.append('"{}": {}'.format(str(name), str(value)))
        statement_params_dict = ', '.join(s_list)

        # generate params statement
        s_list = []
        for name, value in params_list:
            s_list.append('{} = self.params["{}"]'.format(str(name), str(name)))
        statement_params = '\n        '.join(s_list)

        coord = self.coord

        statement_variable_in_coord = self.statement_variable_in_coord
        
        s = self.textEdit_boundary.toPlainText()
        s_list = [line.strip() for line in s.split('\n')]
        statement_boundary = '\n        '.join(s_list)

        statement_shape = self.textEdit_shape.toPlainText().strip()
        
        s = self.textEdit_sld.toPlainText()
        s_list = [line.strip() for line in s.split('\n')]
        statement_sld = '\n        '.join(s_list)

        self.math_model_text = '''
import numpy as np

class specific_mathmodel:

    def __init__(self):
        self.params = {{{0}}}
        self.coord = "{1}"  # 'xyz' or 'sph' or 'cyl'
        self.boundary_min, self.boundary_max = self.genBoundary()
        # must have these 4 attributes

    def genBoundary(self):
        {2}  # params statement

        boundary_min, boundary_max = -1*np.array([1, 1, 1]), np.array([1, 1, 1])
        {3}  # boundary statement
        self.boundary_min, self.boundary_max = boundary_min, boundary_max
        return boundary_min, boundary_max

    def shape(self, grid_in_coord):
        self.grid_in_coord = grid_in_coord
        self.u, self.v, self.w = grid_in_coord[:,0], grid_in_coord[:,1], grid_in_coord[:,2]
        {4} = self.u, self.v, self.w  # u, v, w to certain coordinate variables
        
        # set parameters
        {2}  # params statement

        in_model_grid_index = np.zeros_like(self.u)
        in_model_grid_index[{5}] = 1

        self.in_model_grid_index = in_model_grid_index
        return in_model_grid_index

    def sld(self):
        {4} = self.u, self.v, self.w  # u, v, w to certain coordinate variables

        # set parameters
        {2} # params statement

        sld = np.ones_like(self.u)
        {6}
        
        sld_grid_index = sld * self.in_model_grid_index
        return sld_grid_index
        '''.format(
            statement_params_dict,
            coord,
            statement_params,
            statement_boundary,
            statement_variable_in_coord,
            statement_shape,
            statement_sld
        )

        model_name = self.lineEdit_modelName.text().strip()
        s = 'abcdefghijklmnopqrstuvwxyz'
        allowed_char = s + s.upper() + '0123456789_'
        legal_model_name = ''
        for char in model_name:
            if char in allowed_char:
                legal_model_name += char
            elif char == ' ':
                legal_model_name += '_'
        self.legal_model_name = legal_model_name

    def saveMathModel(self):
        self.genMathModelString()
        folder = QFileDialog.getExistingDirectory(self, 'Select directory to save generated math model', './')
        if folder:
            filename = os.path.join(folder, '{}.py'.format(self.legal_model_name))
            with open(filename, 'w') as f:
                f.write(self.math_model_text)
        

if __name__ == '__main__':
    import sys
    from PyQt5.QtWidgets import QApplication

    app = QApplication(sys.argv)
    window = mathModelGenerationWindow()
    window.show()
    app.exec()