import sys
import os
import math
import tempfile, time
from typing import Literal, Optional

import torch
from PySide6 import QtCore
from PySide6.QtCore import QThread, Signal, QObject
from PySide6.QtGui import QStandardItemModel, QStandardItem
from PySide6.QtWidgets import QApplication, QMainWindow, QWidget, QFileDialog, QInputDialog, QMdiSubWindow, QHeaderView

from .MainWindow_ui import Ui_MainWindow
from .SubWindow_buildmath_ui import Ui_subWindow_build_math_model
from .SubWindow_htmlview_ui import Ui_subWindow_html_view
from .utils import ModelContainer, Project


from ..model import Part, StlPart, MathPart, Assembly
from .. import plot

'''
规划：
1. 目前初期版本只实现一个默认的assembly，不加入多个assembly等功能
'''


class RedirectedPrintStream(QObject):
    
    write_text = Signal(str)
    
    def write(self, text: str):
        self.write_text.emit(text)
        
    def flush(self):
        pass
        
        
class GeneralThread(QThread):
    
    thread_end = Signal()
    
    def __init__(self, func, *args, **kwargs) -> None:
        super().__init__()
        self.func = func
        self.args = args
        self.kwargs = kwargs
        
    def run(self):
        self.func(*self.args, **self.kwargs)
        self.thread_end.emit()
        

class PlotThread(GeneralThread):
    
    thread_end = Signal(str)
    
    def run(self):
        print('ploting')
        if 'savename' in self.kwargs.keys():
            html_filename = self.kwargs['savename']
        else:
            html_filename = os.path.join(tempfile.gettempdir(), 'model2sas_plot.html')
        self.func(*self.args, savename=html_filename, **self.kwargs)
        self.thread_end.emit(html_filename)
        

class MeasureThread(GeneralThread):
    
    thread_end = Signal(tuple)
    
    def run(self):
        I = self.func(*self.args, **self.kwargs)
        q = self.args
        self.thread_end.emit((*q, I))
        

class MainWindow(QMainWindow):
    
    def __init__(self) -> None:
        
        # ui related
        super().__init__()
        
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.show()
        
        # redirect print output and error
        self.system_stdout, self.system_stderr = sys.stdout, sys.stderr
        sys.stdout = RedirectedPrintStream()
        sys.stdout.write_text.connect(self.write_log)
        sys.stderr = RedirectedPrintStream()
        sys.stderr.write_text.connect(self.write_log)
        
        # data related
        self.project = Project()
        self.project.new_assembly('assembly') #* only prelimilary
        self.qmodel_for_treeview = QStandardItemModel()
        self.qmodel_for_model_params_tableview = QStandardItemModel()
        self.ui.treeView_models.setModel(self.qmodel_for_treeview)
        self.ui.tableView_model_params.setModel(self.qmodel_for_model_params_tableview)
        self.ui.tableView_model_params.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.active_model: ModelContainer
        self.thread: QThread
        
        self.build_qtmodel_for_model_treeview()
        
    def write_log(self, text: str):
        if text != '\n': # print() func will print a \n afterwards
            self.ui.textBrowser_log.append(text)
        self.system_stdout.write(text)
        
        
    def load_model_files(self):
        filename_list, _ = QFileDialog.getOpenFileNames(self, caption='Select Model File(s)', dir='./', filter="All Files (*);;stl Files (*.stl);;math model Files (*.py)")
        # filename_list = [
        #     r'D:\Research\my_programs\Model2SAS\resources\exp_models\torus.stl',
        #     r'D:\Research\my_programs\Model2SAS\resources\exp_models\cylinder.py',
        #     r'D:\Research\my_programs\Model2SAS\resources\exp_models\sphere.py',
        # ]
        # print(filename_list)
        loaded_part_keys = []
        for filename in filename_list:
            part_key = self.project.load_part_from_file(filename)
            loaded_part_keys.append(part_key)
        
        #* only prelimilary
        first_assembly_key = list(self.project.assemblies.keys())[0]
        for part_key in loaded_part_keys:
            self.project.add_part_to_assembly(part_key, first_assembly_key)
        #*
        
        self.build_qtmodel_for_model_treeview()
        
    
    def build_qtmodel_for_model_treeview(self):
        # update treeview
        self.qmodel_for_treeview.clear()
        self.qmodel_for_treeview.setHorizontalHeaderLabels(['Models', 'Info'])
        
        def add_root_item(key: str, model: ModelContainer) -> QStandardItem:
            qitem_root = QStandardItem(key)
            self.qmodel_for_treeview.appendRow(qitem_root)
            self.qmodel_for_treeview.setItem(
                self.qmodel_for_treeview.indexFromItem(qitem_root).row(),
                1,
                QStandardItem('{} @{}'.format(model.model.model_type, model.model.device))
            )
            return qitem_root
            
        def add_child_item(root_item: QStandardItem, index: int, key: str, model: ModelContainer):
            qitem_child = QStandardItem(key)
            root_item.appendRow(qitem_child)
            root_item.setChild(
                index,
                1,
                QStandardItem('{} @{}'.format(model.model.model_type, model.model.device))
            )
            return qitem_child
        
        print(self.project.assemblies)
        # top level
        for key, model in self.project.parts.items():
            if len(model.parent) == 0:
                add_root_item(key, model)
        for key, model in self.project.assemblies.items():
            qitem_root = add_root_item(key, model)
            for i, child_part_key in enumerate(model.children):
                add_child_item(
                    qitem_root,
                    i,
                    child_part_key,
                    self.project.parts[child_part_key]
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
        # print(self.ui.treeView_models.selectedIndexes()[0].data())
        selected_key = self.ui.treeView_models.selectedIndexes()[0].data()
        self.ui.label_active_model.setText('Active Model: {}'.format(selected_key))
        if selected_key in self.project.parts.keys():
            # selected part
            self.active_model = self.project.parts[selected_key]
            self.ui.tableView_model_params.setDisabled(False)
            self.ui.pushButton_sampling.setDisabled(False)
            self.ui.pushButton_sampling.setText('Sampling')
            self.ui.pushButton_plot_model.setDisabled(False)
            self.ui.pushButton_scattering.setDisabled(False)
            self.ui.pushButton_scattering.setText('Virtual Scattering')
            self.ui.pushButton_1d_measure.setDisabled(False)
            self.build_qtmodel_for_model_params_tableview()
        else:
            # selected assembly
            self.active_model = self.project.assemblies[selected_key]
            self.qmodel_for_model_params_tableview.clear()
            self.ui.tableView_model_params.setDisabled(True)
            self.ui.pushButton_sampling.setDisabled(False)
            self.ui.pushButton_sampling.setText('Sampling All Sub-Parts')
            self.ui.pushButton_plot_model.setDisabled(False)
            self.ui.pushButton_scattering.setDisabled(False)
            self.ui.pushButton_scattering.setText('Virtual Scattering All Sub-Parts')
            self.ui.pushButton_1d_measure.setDisabled(False)
        self.display_model_settings()
        
    def build_qtmodel_for_model_params_tableview(self):
        self.qmodel_for_model_params_tableview.clear()
        self.qmodel_for_model_params_tableview.setHorizontalHeaderLabels(['Param', 'Value'])
        
        if isinstance(self.active_model.model, StlPart):
            params = dict(sld_value=self.active_model.model.sld_value)
        elif isinstance(self.active_model.model, MathPart):
            params = self.active_model.model.get_params()
        else:
            params = dict()
        
        for i, tp in enumerate(params.items()):
            param_name, param_value = tp
            qitem_param_name = QStandardItem(param_name)
            self.qmodel_for_model_params_tableview.setItem(i, 0, qitem_param_name)
            self.qmodel_for_model_params_tableview.setItem(i, 1, QStandardItem(str(param_value)))
            # set param name not editable
            qitem_param_name.setFlags(QtCore.Qt.ItemFlag(0))
            qitem_param_name.setFlags(QtCore.Qt.ItemIsEnabled)
            
    def read_params_from_tableview_qmodel(self):
        if isinstance(self.active_model.model, StlPart):
            sld_value = float(self.qmodel_for_model_params_tableview.index(0, 1).data())
            self.active_model.model.sld_value = sld_value
        elif isinstance(self.active_model.model, MathPart):
            for i in range(self.qmodel_for_model_params_tableview.rowCount()):
                param_name = self.qmodel_for_model_params_tableview.index(i, 0).data()
                param_value = float(self.qmodel_for_model_params_tableview.index(i, 1).data())
                self.active_model.model.math_model.params[param_name] = param_value
        
        
    def display_model_settings(self):
        self.ui.lineEdit_real_lattice_1d_size.setText(str(self.active_model.real_lattice_1d_size))
        self.ui.lineEdit_q1d_min.setText(str(self.active_model.q1d_min))
        self.ui.lineEdit_q1d_max.setText(str(self.active_model.q1d_max))
        self.ui.lineEdit_q1d_num.setText(str(self.active_model.q1d_num))
        self.ui.checkBox_q1d_log_spaced.setChecked(self.active_model.q1d_log_spaced)
        
        
        if self.active_model.type == 'part':
            self.qmodel_for_params = QStandardItemModel()
            self.qmodel_for_params.setHorizontalHeaderLabels(['Param', 'Value'])
        
    def sampling(self):
        self.read_params_from_tableview_qmodel()
        real_lattice_1d_size = int(self.ui.lineEdit_real_lattice_1d_size.text())
        self.active_model.real_lattice_1d_size = real_lattice_1d_size
        
        if self.active_model.type == 'part':
            self.thread = GeneralThread(
                self.active_model.model.sampling,
                real_lattice_1d_size=real_lattice_1d_size
            )
        else:
            def assembly_sampling(
                assembly: ModelContainer,
                parts: dict[str, ModelContainer],
                real_lattice_1d_size: int
                ):
                for part_key in assembly.children:
                    # print(part_key)
                    part = parts[part_key]
                    part.real_lattice_1d_size = real_lattice_1d_size
                    part.model.sampling(real_lattice_1d_size=real_lattice_1d_size)
            self.thread = GeneralThread(
                assembly_sampling,
                self.active_model,
                self.project.parts,
                real_lattice_1d_size
            )
        self.thread.thread_end.connect(self.sampling_thread_end)
        self.thread.start()
            
    def sampling_thread_end(self):
        # print('sampling done')
        # self.plot_model()
        pass
        
    def rebuild_parts_in_assembly(self, assembly_model: ModelContainer):
        #* rebuild assembly every time used
        if assembly_model.type == 'assembly':
            assembly_model.model.parts = [
                self.project.parts[key].model for key in self.active_model.children
            ]
        
    def plot_model(self):
        if self.active_model.type == 'assembly':
            #* rebuild assembly every time used
            self.rebuild_parts_in_assembly(self.active_model)
        
        self.thread = PlotThread(
            plot.plot_model,
            self.active_model.model,
            title='Model Plot: {}'.format(self.active_model.key),
            show=False,
        )
        self.thread.thread_end.connect(self.display_html)
        self.thread.start()
        
    def display_html(self, html_filename: str):
        # * Known issues
        # * (Solved) must use forward slashes in file path, or will be blank or error
        # * (Solved) begin size can't be too small or plot will be blank
        subwindow_html_view = SubWindowHtmlView()
        # subwindow_html_view.setWindowTitle('Plot Model: {}'.format(self.active_model.key))
        subwindow_html_view.setWindowTitle('Plot')
        subwindow_html_view.ui.webEngineView.setUrl(html_filename.replace('\\', '/'))
        self.ui.mdiArea.addSubWindow(subwindow_html_view)
        subwindow_html_view.show()
        
        
    def virtual_scattering(self):
        if self.active_model.type == 'part':
            # self.active_model.model.scatter()
            self.thread = GeneralThread(self.active_model.model.scatter)
        else:
            # for part_key in self.active_model.children:
            #     self.project.parts[part_key].model.scatter()
            def assembly_scatter(
                assembly: ModelContainer,
                parts: dict[str, ModelContainer],
                ):
                for part_key in assembly.children:
                    # print(part_key)
                    part = parts[part_key]
                    part.model.scatter()
            self.thread = GeneralThread(assembly_scatter, self.active_model, self.project.parts)
        self.thread.thread_end.connect(self.virtual_scattering_thread_end)
        self.thread.start()
                
    def virtual_scattering_thread_end(self):
        print('virtual scattering done')
        
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
            
        if self.active_model.type == 'assembly':
            self.rebuild_parts_in_assembly(self.active_model)
        
        self.thread = MeasureThread(
            self.active_model.model.measure,
            q1d
        )
        self.thread.thread_end.connect(self.measure_thread_end)
        self.thread.start()
        
    def measure_thread_end(self, qI: tuple[torch.Tensor]):
        q, I = qI[:-1], qI[-1]
        if len(q) == 1:
            self.thread = PlotThread(
                plot.plot_1d_sas,
                q[0],
                I,
                mode='lines',
                name=self.active_model.key,
                title=self.active_model.key,
                show=False,
            )
        else:
            self.thread = PlotThread(
                plot.plot_2d_sas,
                I,
                logI=True,
                title=self.active_model.key,
                show=False,
            )
        self.thread.thread_end.connect(self.display_html)
        self.thread.start()
        
        
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
        