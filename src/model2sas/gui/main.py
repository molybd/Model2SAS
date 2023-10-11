import sys
import os
import math
import tempfile, time
from typing import Literal, Optional, Union

import torch
from PySide6 import QtCore
from PySide6.QtCore import QThread, Signal, Slot, QObject, QModelIndex, SIGNAL, SLOT
from PySide6.QtGui import QStandardItemModel, QStandardItem
from PySide6.QtWidgets import QApplication, QMainWindow, QWidget, QFileDialog, QInputDialog, QMdiSubWindow, QHeaderView, QStyledItemDelegate, QComboBox
from art import text2art

from .MainWindow_ui import Ui_MainWindow
from .SubWindow_buildmath_ui import Ui_subWindow_build_math_model
from .SubWindow_htmlview_ui import Ui_subWindow_html_view
from .wrapper import Project, StlPartModel, MathPartModel, AssemblyModel, PartModel


from ..model import Part, StlPart, MathPart, Assembly
from .. import plot

from ..utils import logger, set_log_state, LOG_FORMAT_STR

'''
TODO
1. BUG 一次导入多个模型会显示不全（没有复现）
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
        

class TransformTypeComboDelegate(QStyledItemDelegate):
    """refer to:
    https://stackoverflow.com/questions/17615997/pyqt-how-to-set-qcombobox-in-a-table-view-using-qitemdelegate
    """
    def __init__(self, parent: QObject | None = ...) -> None:
        super().__init__(parent)
        
    def createEditor(self, parent, option, index) -> QWidget:
        combo  = QComboBox(parent)
        combo.addItems(['Translate', 'Rotate'])
        self.connect(combo, SIGNAL('currentIndexChanged(int)'), self, SLOT('currentIndexChanged()'))
        return combo
    
    def setEditorData(self, editor: QComboBox, index: QModelIndex) -> None:
        editor.blockSignals(True)
        # index.data() is actually index string, like "0" and "1"
        # not actually displayed values
        editor.setCurrentIndex(int(index.data()))
        editor.blockSignals(False)
        
    def setModelData(self, editor: QComboBox, model: QStandardItemModel, index: QModelIndex) -> None:
        model.setData(index, editor.currentIndex())
    
    @Slot()
    def currentIndexChanged(self):
        self.commitData.emit(self.sender())


class MainWindow(QMainWindow):
    
    def __init__(self) -> None:
        # ui related
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.show()
        
        # redirect print output and error
        # self.system_stdout, self.system_stderr = sys.stdout, sys.stderr
        # sys.stdout = RedirectedPrintStream()
        # sys.stdout.write_text.connect(self.write_log)
        # sys.stderr = RedirectedPrintStream()
        # sys.stderr.write_text.connect(self.write_log)
        
        # logger.remove(0)
        logger.add(self.write_log, format=LOG_FORMAT_STR)
        set_log_state(True)
        
        # self.gui_log('success', 'test')
        # self.gui_log('info', 'test')
        # self.gui_log('warning', 'test')
        # self.gui_log('error', 'test')
        
        # set model-view
        self.qitemmodel_parts = QStandardItemModel()
        self.ui.treeView_parts.setModel(self.qitemmodel_parts)
        
        self.qitemmodel_assmblies = QStandardItemModel()
        self.ui.treeView_assemblies.setModel(self.qitemmodel_assmblies)
        
        self.qitemmodel_params = QStandardItemModel()
        self.qitemmodel_params.itemChanged.connect(self.read_params_from_qitemmodel)
        self.ui.tableView_model_params.setModel(self.qitemmodel_params)
        
        self.qitemmodel_transforms = QStandardItemModel()
        # self.qitemmodel_transforms.itemChanged.connect(self.read_transforms_from_qitemmodel)
        self.ui.tableView_transforms.setModel(self.qitemmodel_transforms)
        self.ui.tableView_transforms.setItemDelegateForColumn(0, TransformTypeComboDelegate(self))
                
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
        
        # welcome message
        welcome_message = text2art('Model2SAS')
        welcome_message += text2art('Small angle scattering simulation from 3d models', font='fancy1', decoration='barcode1') + '\n\n'
        welcome_message += '🏠️ Homepage: https://github.com/molybd/Model2SAS\n'
        welcome_message += '📄 Please site: Li, Mu and Yin, Panchao, Model2SAS: software for small-angle scattering data calculation from custom shapes., J. Appl. Cryst., 2022, 55, 663-668. https://doi.org/10.1107/S1600576722003600\n'
        self.ui.textBrowser_log.append(welcome_message)
        print(welcome_message)
        
        
        
    def write_log(self, text: str):
        self.ui.textBrowser_log.append(text.strip()) # use .strip(), or will have blank line
        
        # scroll textBrowser_log to bottom
        cursor = self.ui.textBrowser_log.textCursor()  # 设置游标
        pos = len(self.ui.textBrowser_log.toPlainText())  # 获取文本尾部的位置
        cursor.setPosition(pos)  # 游标位置设置为尾部
        self.ui.textBrowser_log.setTextCursor(cursor)  # 滚动到游标位置
        
        
    def gui_log(self, type: Literal['success', 'info', 'warning', 'error'], text: str):
        if type == 'info':
            func = logger.info
            symbol = 'ℹ️'
        elif type == 'warning':
            func = logger.warning
            symbol = '⚠️'
        elif type == 'error':
            func = logger.error
            symbol = '❌️'
        else:
            func = logger.success
            symbol = '✅️'
        func(f'[{" ":>11}] {symbol} {text}')
    
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
        for filename in filename_list:
            self.project.import_part(filename)
            self.gui_log('info', f'imported {filename}')
            
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
        self.active_model = self.project.parts[selected_key]
        self.gui_log('info', f'select 【{selected_key}】{self.active_model.name}')
        self._unique_actions_for_part_model_selected()
        self._actions_for_model_selected()
    
    def assembly_model_selected(self) -> None:
        selected_key = str(self.ui.treeView_assemblies.selectedIndexes()[0].data())
        if selected_key in self.project.parts.keys():
            # part model selected
            self.active_model = self.project.parts[selected_key]
            self._unique_actions_for_part_model_selected()
        else:
            # assembly model selected
            self.active_model = self.project.assemblies[selected_key]
            self._unique_actions_for_assembly_model_selected()
        self.gui_log('info', f'select 【{selected_key}】{self.active_model.name}')
        self._actions_for_model_selected()
            
    
    def _unique_actions_for_part_model_selected(self) -> None:
        self.ui.pushButton_add_to_assembly.setDisabled(False)
        self.ui.tableView_model_params.setDisabled(False)
        self.ui.pushButton_sample.setDisabled(False)
        self.ui.pushButton_sample.setText('Sample')
        self.ui.pushButton_add_transform.setDisabled(False)
        self.ui.pushButton_delete_selected_transform.setDisabled(False)
        self.ui.pushButton_apply_transform.setDisabled(False)
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
        self.ui.pushButton_apply_transform.setDisabled(True)
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
            self.gui_log('success', 'change parameter')
        elif isinstance(self.active_model, MathPartModel):
            for i in range(self.qitemmodel_params.rowCount()):
                param_name = self.qitemmodel_params.index(i, 0).data()
                param_value = float(self.qitemmodel_params.index(i, 1).data())
                self.active_model.set_params(**{param_name: param_value})
            self.gui_log('success', 'change parameter')
            
    def refresh_qitemmodel_transforms(self) -> None:
        self.qitemmodel_transforms.clear()
        self.qitemmodel_transforms.setHorizontalHeaderLabels(['Type', 'Vector/Axis', 'Angle(deg)'])
        for i, transform in enumerate(self.active_model.model.geo_transform):
            if transform['type'] == 'rotate':
                combo_index = '1'
                vec, angle = transform['param']
                angle = math.degrees(angle)
            else:
                combo_index = '0'
                vec, angle = transform['param'], 'N.A.'
            self.qitemmodel_transforms.setItem(
                i, 0, QStandardItem(combo_index)
            )
            self.qitemmodel_transforms.setItem(
                i, 1, QStandardItem(','.join(map(str, vec)))
            )
            self.qitemmodel_transforms.setItem(
                i, 2, QStandardItem(str(angle))
            )
            self.ui.tableView_transforms.openPersistentEditor(
                self.qitemmodel_transforms.index(i, 0)
            )
            
    def add_transform(self) -> None:
        row_count = self.qitemmodel_transforms.rowCount()
        self.qitemmodel_transforms.setItem(row_count, 0, QStandardItem('0'))
        self.ui.tableView_transforms.openPersistentEditor(
            self.qitemmodel_transforms.index(row_count, 0)
        )
        
    def delete_selected_transform(self) -> None:
        selected_index = self.ui.tableView_transforms.selectedIndexes()[0].row()
        self.qitemmodel_transforms.removeRow(selected_index)   
        
    def apply_transform(self) -> None:
        self.active_model.model.geo_transform.clear() # clear former records first
        for i in range(self.qitemmodel_transforms.rowCount()):
            combo_index = int(self.qitemmodel_transforms.index(i, 0).data())
            vec = tuple(map(float, self.qitemmodel_transforms.index(i, 1).data().split(',')))
            if combo_index == 1: # rotate
                angle = float(self.qitemmodel_transforms.index(i, 2).data())
                angle = math.radians(angle)
                self.active_model.model.rotate(vec, angle)
            else: # translate
                self.active_model.model.translate(*vec)
        self.refresh_qitemmodel_transforms()
            
    def sample(self) -> None:
        self.thread = GeneralThread(
            self.active_model.sample
        )
        self.thread.thread_end.connect(self.sample_thread_end)
        self.thread.start()
                
    def sample_thread_end(self) -> None:
        self.gui_log('success', 'Sample done')
        
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
        self.gui_log('info', 'Plotting...')
        
    def scatter(self) -> None:
        self.thread = GeneralThread(
            self.active_model.scatter
        )
        self.thread.thread_end.connect(self.scatter_thread_end)
        self.thread.start()
                
    def scatter_thread_end(self) -> None:
        self.gui_log('success', 'Scatter done')
        
    def measure(self) -> None:
        self.thread = MeasureThread(
            self.active_model.measure
        )
        self.thread.thread_end.connect(self.measure_thread_end)
        self.thread.start()
        
    def measure_thread_end(self, result: tuple) -> None:
        self.gui_log('success', 'Measure done')
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
        self.gui_log('success', 'Plot done')
    





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