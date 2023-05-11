import sys
import os
import math
import tempfile, time
from typing import Optional

import torch
from PySide6.QtCore import QObject, QThread, Signal
from PySide6.QtGui import QStandardItemModel, QStandardItem
from PySide6.QtWidgets import QApplication, QMainWindow, QWidget, QFileDialog, QInputDialog, QMdiSubWindow

from .MainWindow_ui import Ui_MainWindow
from .SubWindow_buildmath_ui import Ui_subWindow_build_math_model
from .SubWindow_htmlview_ui import Ui_subWindow_html_view
from .utils import PartContainer, AssemblyContainer, Project

from ..model import Part, StlPart, MathPart, Assembly
from .. import plot

'''
规划：
1. 目前初期版本只实现一个默认的assembly，不加入多个assembly等功能
'''


class ConsolePrintStream:
    def __init__(self, text_browser, system_print) -> None:
        self.text_browser = text_browser
        self.system_print = system_print
        
    def write(self, text):
        if text != '\n': # print() func will print a \n afterwards
            self.text_browser.append(text)
        self.system_print.write(text)


class MainWindow(QMainWindow):
    
    def __init__(self, app) -> None:
        
        self.app = app
        # ui related
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.show()
        
        # redirect print output and error
        sys.stdout = ConsolePrintStream(self.ui.textBrowser_log, sys.stdout)
        sys.stderr = ConsolePrintStream(self.ui.textBrowser_log, sys.stderr)
        
        # data related
        self.project = Project()
        self.project.new_assembly()
        self.qmodel_for_treeview = QStandardItemModel()
        self.ui.treeView_models.setModel(self.qmodel_for_treeview)
        self.active_model: PartContainer | AssemblyContainer
        
        self.update_ui_models_tree_changed()
        
        
    def load_model_files(self):
        # filename_list, _ = QFileDialog.getOpenFileNames(self, caption='Select Model File(s)', dir='./', filter="All Files (*);;stl Files (*.stl);;math model Files (*.py)")
        filename_list = [
            r'D:\Research\my_programs\Model2SAS\resources\exp_models\torus.stl',
            r'D:\Research\my_programs\Model2SAS\resources\exp_models\cylinder.py',
            r'D:\Research\my_programs\Model2SAS\resources\exp_models\sphere.py',
        ]
        print(filename_list)
        for filename in filename_list:
            self.project.add_part(filename, 'cpu')
        
        self.update_ui_models_tree_changed()
        
    
    def update_ui_models_tree_changed(self):
        # update treeview
        self.qmodel_for_treeview.clear()
        self.qmodel_for_treeview.setHorizontalHeaderLabels(['Models', 'Info'])
        
        # top level
        model = self.project.assembly.model
        qitem_assembly = QStandardItem('assembly')
        self.qmodel_for_treeview.appendRow(qitem_assembly)
        self.qmodel_for_treeview.setItem(
            self.qmodel_for_treeview.indexFromItem(qitem_assembly).row(),
            1,
            QStandardItem('{} model @{}'.format(model.model_type, model.device))
        )
        for i, item in enumerate(self.project.parts.items()):
            partname, model = item[0], item[1].model
            qitem_part = QStandardItem(partname)
            qitem_assembly.appendRow(qitem_part)
            qitem_assembly.setChild(
                i,
                1,
                QStandardItem('{} model @{}'.format(model.model_type, model.device))
            )
        self.ui.treeView_models.expandAll()
        
        # update combobox_assembly_list
        # self.ui.comboBox_assmbly_list.clear()
        # for key, model in self.model2sas_models.items():
        #     if model.model_type == 'assembly':
        #         self.ui.comboBox_assmbly_list.addItem(key)
        
    def build_math_model(self):
        print('clicked build_math_model')
        subwindow_build_math_model = SubWindowBuildMathModel()
        self.ui.mdiArea.addSubWindow(subwindow_build_math_model)
        subwindow_build_math_model.show()
    
    def selected_model_settings(self):
        if self.ui.treeView_models.selectedIndexes()[0].parent().row() == -1:
            # selected assembly
            # print('selected assembly')
            self.ui.label_active_model.setText('Active Model: assembly')
            self.active_model = self.project.assembly
            self.ui.tableView_model_params.setDisabled(True)
            self.ui.pushButton_sampling.setDisabled(False)
            self.ui.pushButton_sampling.setText('Sampling All Sub-Parts')
            self.ui.pushButton_plot_model.setDisabled(False)
            self.ui.pushButton_scattering.setDisabled(False)
            self.ui.pushButton_scattering.setText('Virtual Scattering All Sub-Parts')
            self.ui.pushButton_1d_measure.setDisabled(False)
        else:
            # selected part
            index = self.ui.treeView_models.selectedIndexes()[0].row()
            partname = self.ui.treeView_models.selectedIndexes()[0].data()
            # print('selected part with index={}'.format(index))
            self.ui.label_active_model.setText('Active Model: {}'.format(partname))
            self.active_model = self.project.parts[partname]
            self.ui.tableView_model_params.setDisabled(False)
            self.ui.pushButton_sampling.setDisabled(False)
            self.ui.pushButton_sampling.setText('Sampling')
            self.ui.pushButton_plot_model.setDisabled(False)
            self.ui.pushButton_scattering.setDisabled(False)
            self.ui.pushButton_scattering.setText('Virtual Scattering')
            self.ui.pushButton_1d_measure.setDisabled(False)
        self.display_model_settings()
            
    def display_model_settings(self):
        self.ui.lineEdit_real_lattice_1d_size.setText(str(self.active_model.real_lattice_1d_size))
        self.ui.lineEdit_q1d_min.setText(str(self.active_model.q1d_min))
        self.ui.lineEdit_q1d_max.setText(str(self.active_model.q1d_max))
        self.ui.lineEdit_q1d_num.setText(str(self.active_model.q1d_num))
        self.ui.checkBox_q1d_log_spaced.setChecked(self.active_model.q1d_log_spaced)
        
        
        if isinstance(self.active_model, PartContainer):
            self.qmodel_for_params = QStandardItemModel()
            self.qmodel_for_params.setHorizontalHeaderLabels(['Param', 'Value'])
        
        
            
    def sampling(self):
        
        def part_sampling(part: PartContainer):
            real_lattice_1d_size = int(self.ui.lineEdit_real_lattice_1d_size.text())
            part.real_lattice_1d_size = real_lattice_1d_size
            part.model.sampling(real_lattice_1d_size=real_lattice_1d_size)
        
        if isinstance(self.active_model, AssemblyContainer):
            for part in self.active_model.model.parts:
                part_sampling(self.project.parts[part.partname])
            self.active_model.real_lattice_1d_size = int(self.ui.lineEdit_real_lattice_1d_size.text())
        else:
            part_sampling(self.active_model)
        self.plot_model()
        
    def plot_model(self):
        html_filename = os.path.join(tempfile.gettempdir(), 'model2sas_plot.html'.format(time.time()))
        # html_filename = './temp.html'
        plot.plot_model(
            self.active_model.model,
            title=self.active_model.key,
            show=False,
            savename=html_filename
        )
        # * Known issues
        # * (Solved) must use forward slashes in file path, or will be blank or error
        # * (Solved) begin size can't be too small or plot will be blank
        subwindow_html_view = SubWindowHtmlView()
        subwindow_html_view.setWindowTitle('Plot Model: {}'.format(self.active_model.key))
        subwindow_html_view.ui.webEngineView.setUrl(html_filename.replace('\\', '/'))
        self.ui.mdiArea.addSubWindow(subwindow_html_view)
        subwindow_html_view.show()
        
    def virtual_scattering(self):
        if isinstance(self.active_model, AssemblyContainer):
            for part in self.active_model.model.parts:
                part.scatter()
        else:
            self.active_model.model.scatter()
        
    def measure_1d(self):
        q1d_min = float(self.ui.lineEdit_q1d_min.text())
        q1d_max = float(self.ui.lineEdit_q1d_max.text())
        q1d_num = int(self.ui.lineEdit_q1d_num.text())
        q1d_log_spaced = self.ui.checkBox_q1d_log_spaced.isChecked()
        self.active_model.q1d_min = q1d_min
        self.active_model.q1d_max = q1d_max
        self.active_model.q1d_num = q1d_num
        self.active_model.q1d_log_spaced = q1d_log_spaced
        
        if q1d_log_spaced:
            q1d = torch.logspace(
                math.log10(q1d_min),
                math.log10(q1d_max),
                steps=q1d_num
            )
        else:
            q1d = torch.linspace(q1d_min, q1d_max, steps=q1d_num)
        I1d = self.active_model.model.measure(q1d)
        
        html_filename = os.path.join(tempfile.gettempdir(), 'model2sas_plot.html'.format(time.time()))
        # html_filename = './temp.html'
        plot.plot_1d_sas(
            q1d,
            I1d,
            mode='lines',
            name=self.active_model.key,
            title=self.active_model.key,
            show=False,
            savename=html_filename
        )
        # * Known issues
        # * (Solved) must use forward slashes in file path, or will be blank or error
        # * (Solved) begin size can't be too small or plot will be blank
        subwindow_html_view = SubWindowHtmlView()
        subwindow_html_view.setWindowTitle('Plot Scattering: {}'.format(self.active_model.key))
        subwindow_html_view.ui.webEngineView.setUrl(html_filename.replace('\\', '/'))
        self.ui.mdiArea.addSubWindow(subwindow_html_view)
        subwindow_html_view.show()
        
        
        
class SubWindowBuildMathModel(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.ui = Ui_subWindow_build_math_model()
        self.ui.setupUi(self)
        
class SubWindowHtmlView(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.ui = Ui_subWindow_html_view()
        self.ui.setupUi(self)
        