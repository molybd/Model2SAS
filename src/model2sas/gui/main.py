import sys
import os
import math
import tempfile, time
from typing import Literal, Optional

import torch
from PySide6 import QtCore
from PySide6.QtCore import QThread, Signal, QObject, QModelIndex
from PySide6.QtGui import QStandardItemModel, QStandardItem
from PySide6.QtWidgets import QApplication, QMainWindow, QWidget, QFileDialog, QInputDialog, QMdiSubWindow, QHeaderView

from .MainWindow_ui import Ui_MainWindow
from .SubWindow_buildmath_ui import Ui_subWindow_build_math_model
from .SubWindow_htmlview_ui import Ui_subWindow_html_view
from .wrapper import Project, StlPartModel, MathPartModel, AssemblyModel, PartModel


from ..model import Part, StlPart, MathPart, Assembly
from .. import plot

'''
TODO

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
        

class MeasureThread(GeneralThread):
    thread_end = Signal(tuple)
    def run(self):
        result = self.func(*self.args, **self.kwargs)
        self.thread_end.emit(result)
        

class PlotThread(GeneralThread):
    thread_end = Signal(str)
    def run(self):
        # print('ploting')
        if 'savename' in self.kwargs.keys():
            html_filename = self.kwargs.pop('savename')
        else:
            html_filename = os.path.join(tempfile.gettempdir(), 'model2sas_temp_plot.html')
        self.func(*self.args, savename=html_filename, **self.kwargs)
        self.thread_end.emit(html_filename)


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
        
        # set model-view
        self.qitemmodel_parts = QStandardItemModel()
        self.qitemmodel_assmblies = QStandardItemModel()
        self.qitemmodel_params = QStandardItemModel()
        self.qitemmodel_transforms = QStandardItemModel()
        self.ui.treeView_parts.setModel(self.qitemmodel_parts)
        self.ui.treeView_assemblies.setModel(self.qitemmodel_assmblies)
        self.ui.tableView_model_params.setModel(self.qitemmodel_params)
        self.ui.listView_transforms.setModel(self.qitemmodel_transforms)
        self.qitemmodel_params.itemChanged.connect(self.read_params_from_qitemmodel)
        
        # other variables
        self.active_model: StlPartModel | MathPartModel | AssemblyModel | PartModel
        self.thread: QThread
        
        # link setting values to lineEdit contents      
        self.ui.lineEdit_real_lattice_1d_size.textChanged.connect(self.change_variable_real_lattice_1d_size)
        self.ui.lineEdit_q1d_min.textChanged.connect(self.change_variable_q1d_min)
        self.ui.lineEdit_q1d_max.textChanged.connect(self.change_variable_q1d_max)
        self.ui.lineEdit_q1d_num.textChanged.connect(self.change_variable_q1d_num)
        self.ui.checkBox_q1d_log_spaced.clicked.connect(self.change_variable_q1d_log_spaced)        
        
        # initial actions
        self.project = Project()
        # self.project.new_assembly() #* only prelimilary
        
        
    def write_log(self, text: str):
        if text != '\n': # print() func will print a \n afterwards
            self.ui.textBrowser_log.append(text)
        self.system_stdout.write(text)
    
    ###### * Link setting values to lineEdit contents * ######
    def change_variable_real_lattice_1d_size(self, value: str):
        self.active_model.real_lattice_1d_size = int(float(value))
    def change_variable_q1d_min(self, value: str):
        self.active_model.q1d_min = float(value)
    def change_variable_q1d_max(self, value: str):
        self.active_model.q1d_max = float(value)
    def change_variable_q1d_num(self, value: str):
        self.active_model.q1d_num = int(float(value))
    def change_variable_q1d_log_spaced(self, state: int):
        print(state)
        self.active_model.q1d_log_spaced = bool(state)
    
    ##########################################################*
        
    def import_parts(self) -> None:
        filename_list, _ = QFileDialog.getOpenFileNames(self, caption='Select Model File(s)', dir='./', filter="All Files (*);;stl Files (*.stl);;math model Files (*.py)")
        #* for test only
        # filename_list = [
        #     r'D:\Work@IASF\@my_programs\Model2SAS\resources\exp_models\torus.stl',
        #     r'D:\Work@IASF\@my_programs\Model2SAS\resources\exp_models\cylinder.py',
        #     r'D:\Work@IASF\@my_programs\Model2SAS\resources\exp_models\sphere.py',
        # ]
        #*
        print(filename_list)
        for filename in filename_list:
            self.project.import_part(filename)
            
        #* only prelimilary
        # for part in self.project.parts.values():
        #     self.project.add_part_to_assembly(
        #         part, list(self.project.assemblies.values())[0]
        #     )
        #*
        
        self.refresh_qitemmodel_models()
        
    def add_to_assembly(self) -> None:
        assembly_key = self.ui.comboBox_assemblies.currentText()
        self.project.add_part_to_assembly(self.active_model.key, assembly_key)
        self.refresh_qitemmodel_models()
        
    def new_assembly(self) -> None:
        self.project.new_assembly()
        self.refresh_qitemmodel_models()
        self.refresh_combobox_assemblies()
        
    def refresh_combobox_assemblies(self) -> None:
        self.ui.comboBox_assemblies.clear()
        self.ui.comboBox_assemblies.addItems(self.project.assemblies.keys())
        
    def delete_selected_model(self) -> None:
        if self.active_model in self.project.parts.values():
            # part model selected
            del self.project.parts[self.active_model.key]
            for assembly in self.project.assemblies.values():
                if self.active_model in assembly.parts:
                    assembly.parts.remove(self.active_model)
        else:
            # assembly model selected
            del self.project.assemblies[self.active_model.key]
        self.refresh_qitemmodel_models()
        self.refresh_combobox_assemblies()
        
    def refresh_qitemmodel_models(self) -> None:
        self.qitemmodel_parts.clear()
        self.qitemmodel_parts.setHorizontalHeaderLabels(['ID', 'Name'])
        for part in self.project.parts.values():
            self.qitemmodel_parts.appendRow([
                QStandardItem(part.key),
                QStandardItem(part.name),
            ])
            
        self.qitemmodel_assmblies.clear()
        self.qitemmodel_assmblies.setHorizontalHeaderLabels(['ID', 'Name'])
        for assembly in self.project.assemblies.values():
            root_item = QStandardItem(assembly.key)
            self.qitemmodel_assmblies.appendRow([
                root_item,
                QStandardItem(assembly.name)
            ])
            for part in assembly.parts:
                root_item.appendRow([
                    QStandardItem(part.key),
                    QStandardItem(part.name)
                ])
                
        self.ui.treeView_parts.expandAll()
        self.ui.treeView_assemblies.expandAll()
        
    def part_model_selected(self) -> None:
        selected_key = str(self.ui.treeView_parts.selectedIndexes()[0].data())
        # print(selected_key)
        self.active_model = self.project.parts[selected_key]
        self._unique_actions_for_part_model_selected()
        self._actions_for_model_selected()
    
    def assembly_model_selected(self) -> None:
        selected_key = str(self.ui.treeView_assemblies.selectedIndexes()[0].data())
        print(selected_key)
        if selected_key in self.project.parts.keys():
            # part model selected
            self.active_model = self.project.parts[selected_key]
            self._unique_actions_for_part_model_selected()
        else:
            # assembly model selected
            self.active_model = self.project.assemblies[selected_key]
            self._unique_actions_for_assembly_model_selected()
        self._actions_for_model_selected()
            
    
    def _unique_actions_for_part_model_selected(self) -> None:
        self.ui.pushButton_add_to_assembly.setDisabled(False)
        self.ui.tableView_model_params.setDisabled(False)
        self.ui.pushButton_sample.setDisabled(False)
        self.ui.pushButton_sample.setText('Sample')
        self.ui.pushButton_add_transform.setDisabled(False)
        self.ui.pushButton_delete_selected_transform.setDisabled(False)
        self.ui.pushButton_plot_model.setDisabled(False)
        self.ui.pushButton_scatter.setDisabled(False)
        self.ui.pushButton_scatter.setText('Virtual Scatter')
        self.ui.pushButton_1d_measure.setDisabled(False)
        self.refresh_qitemmodel_params()
        self.refresh_qitemmodel_transforms()
        
        
    def _unique_actions_for_assembly_model_selected(self) -> None:
        self.ui.pushButton_add_to_assembly.setDisabled(True)
        self.ui.tableView_model_params.setDisabled(True)
        self.ui.pushButton_sample.setDisabled(False)
        self.ui.pushButton_sample.setText('Sample All Sub-Parts')
        self.ui.pushButton_add_transform.setDisabled(True)
        self.ui.pushButton_delete_selected_transform.setDisabled(True)
        self.ui.pushButton_plot_model.setDisabled(False)
        self.ui.pushButton_scatter.setDisabled(False)
        self.ui.pushButton_scatter.setText('Virtual Scatter by All Sub-Parts')
        self.ui.pushButton_1d_measure.setDisabled(False)
        self.qitemmodel_params.clear()
        self.qitemmodel_transforms.clear()
    
    def _actions_for_model_selected(self) -> None:
        self.ui.label_active_model.setText(
            f'Active Model: 【{self.active_model.key}】 {self.active_model.name}'
        )
        self.ui.lineEdit_real_lattice_1d_size.setText(str(self.active_model.real_lattice_1d_size))
        self.ui.lineEdit_q1d_min.setText(str(self.active_model.q1d_min))
        self.ui.lineEdit_q1d_max.setText(str(self.active_model.q1d_max))
        self.ui.lineEdit_q1d_num.setText(str(self.active_model.q1d_num))
        self.ui.checkBox_q1d_log_spaced.setChecked(self.active_model.q1d_log_spaced)
        
    def refresh_qitemmodel_params(self) -> None:
        self.qitemmodel_params.clear()
        self.qitemmodel_params.setHorizontalHeaderLabels(['Param', 'Value'])
        if isinstance(self.active_model, StlPartModel):
            params = dict(sld_value=self.active_model.sld_value)
        elif isinstance(self.active_model, MathPartModel):
            params = self.active_model.get_params()
        else:
            params = dict()
        for param_name, param_value in params.items():
            # set param name not editable
            qitem_param_name = QStandardItem(param_name)
            qitem_param_name.setFlags(QtCore.Qt.ItemFlag(0))
            qitem_param_name.setFlags(QtCore.Qt.ItemIsEnabled)
            self.qitemmodel_params.appendRow([
                qitem_param_name,
                QStandardItem(str(param_value))
            ])
            
    def read_params_from_qitemmodel(self):
        if isinstance(self.active_model, StlPartModel):
            self.active_model.sld_value = \
                float(self.qitemmodel_params.index(0, 1).data())
            print(self.active_model.sld_value)
        elif isinstance(self.active_model, MathPartModel):
            for i in range(self.qitemmodel_params.rowCount()):
                param_name = self.qitemmodel_params.index(i, 0).data()
                param_value = float(self.qitemmodel_params.index(i, 1).data())
                self.active_model.set_params(**{param_name: param_value})
            print(self.active_model.get_params())
            
    def refresh_qitemmodel_transforms(self) -> None:
        self.qitemmodel_transforms.clear()
        for transform in self.active_model.model.geo_transform:
            self.qitemmodel_transforms.appendRow(
                QStandardItem(f"{transform['type']} with param: {transform['param']}")
            )
            
    def add_transform(self) -> None:
        transform_type = self.ui.comboBox_transform_type.currentText().lower()
        vec = [float(i) for i in self.ui.lineEdit_transform_vector.text().split(',')]
        if transform_type == 'translate':
            self.active_model.model.translate(*vec[:3])
        elif transform_type == 'rotate':
            angle = math.radians(float(self.ui.lineEdit_transform_angle.text()))
            self.active_model.model.rotate(tuple(vec[:3]), angle)
        self.refresh_qitemmodel_transforms()
        
    def delete_selected_transform(self) -> None:
        selected_index = self.ui.listView_transforms.selectedIndexes()[0].row()
        del self.active_model.model.geo_transform[selected_index]
        self.refresh_qitemmodel_transforms()        
            
    def sample(self) -> None:
        self.thread = GeneralThread(
            self.active_model.sample
        )
        self.thread.thread_end.connect(self.sample_thread_end)
        self.thread.start()
                
    def sample_thread_end(self) -> None:
        print('Sampling Done')
        
    def plot_model(self) -> None:
        if self.ui.radioButton_voxel_plot.isChecked():
            plot_type = 'voxel'
        elif self.ui.radioButton_volume_plot.isChecked():
            plot_type = 'volume'
        if self.active_model.type == 'assembly':
            model = self.active_model.model.parts
        else:
            model = [self.active_model.model]
        self.thread = PlotThread(
            plot.plot_model,
            *model,
            type = plot_type,
            title=f'Model Plot: {self.active_model.key}',
            show=False,
        )
        self.thread.thread_end.connect(self.display_html)
        self.thread.start()
        
    def scatter(self) -> None:
        self.thread = GeneralThread(
            self.active_model.scatter
        )
        self.thread.thread_end.connect(self.scatter_thread_end)
        self.thread.start()
                
    def scatter_thread_end(self) -> None:
        print('Scattering Done')
        
    def measure(self) -> None:
        self.thread = MeasureThread(
            self.active_model.measure
        )
        self.thread.thread_end.connect(self.measure_thread_end)
        self.thread.start()
        
    def measure_thread_end(self, result: tuple) -> None:
        print('Measuring Done')
        q, I = result
        self.thread = PlotThread(
            plot.plot_1d_sas,
            q,
            I,
            mode = 'lines',
            name = f'【{self.active_model.key}】 {self.active_model.name}',
            title = f'【{self.active_model.key}】 {self.active_model.name}',
            show = False
        )
        self.thread.thread_end.connect(self.display_html)
        self.thread.start()
        
        
    def display_html(self, html_filename: str) -> None:
        # * Known issues
        # * (Solved) must use forward slashes in file path, or will be blank or error
        # * (Solved) begin size can't be too small or plot will be blank
        subwindow_html_view = SubWindowHtmlView()
        subwindow_html_view.setWindowTitle('Plot')
        subwindow_html_view.ui.webEngineView.setUrl(html_filename.replace('\\', '/'))
        self.ui.mdiArea.addSubWindow(subwindow_html_view)
        subwindow_html_view.show()
    





class SubWindowHtmlView(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.ui = Ui_subWindow_html_view()
        self.ui.setupUi(self)
        
        
def run():
    app = QApplication(sys.argv)
    main_window = MainWindow()
    sys.exit(app.exec())

if __name__ == '__main__':
    run()