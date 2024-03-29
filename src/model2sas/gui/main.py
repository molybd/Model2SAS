import sys
import os
import math
import tempfile, time
from typing import Literal, Optional, Union

from torch import Tensor
from PySide6 import QtCore
from PySide6.QtCore import QThread, Signal, Slot, QObject, QModelIndex, SIGNAL, SLOT
from PySide6.QtGui import QStandardItemModel, QStandardItem
from PySide6.QtWidgets import QApplication, QMainWindow, QWidget, QFileDialog, QInputDialog, QMdiSubWindow, QHeaderView, QStyledItemDelegate, QComboBox, QLineEdit, QCheckBox, QGridLayout, QPushButton, QSpacerItem, QSizePolicy
from PySide6.QtWebEngineWidgets import QWebEngineView
from art import text2art
import plotly.graph_objects as go

from .mainwindow_ui import Ui_MainWindow
# from .subwindow_htmlview_ui import Ui_subWindow_html_view
from .model_wrapper import Project, StlPartWrapper, MathPartWrapper, AssemblyWrapper, PartWrapper, ModelWrapperType


from ..model import Part, StlPart, MathPart, Assembly
from .. import plot
from ..utils import logger, set_log_state, LOG_FORMAT_STR, WELCOME_MESSAGE
from .utils import GeneralThread, MeasureThread, PlotThread
from .subwindows import SubWindowHtmlView, SubWindowPlotView, SubWindowUserDefinedModel


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
        
        # logger.remove(0)
        logger.add(self.write_log, format=LOG_FORMAT_STR)
        set_log_state(True)
        
        # set model-view
        self.qitemmodel_parts = QStandardItemModel()
        self.ui.treeView_parts.setModel(self.qitemmodel_parts)
        
        self.qitemmodel_assmblies = QStandardItemModel()
        self.ui.treeView_assemblies.setModel(self.qitemmodel_assmblies)
        
        self.qitemmodel_params = QStandardItemModel()
        self.qitemmodel_params.itemChanged.connect(self.read_params_from_qitemmodel)
        self.ui.tableView_model_params.setModel(self.qitemmodel_params)
        
        self.qitemmodel_transforms = QStandardItemModel()
        self.ui.tableView_transforms.setModel(self.qitemmodel_transforms)
        self.ui.tableView_transforms.setItemDelegateForColumn(0, TransformTypeComboDelegate(self))
                
        # other variables
        self.active_model: ModelWrapperType
        self.thread: QThread    
        
        # initial actions
        self.project = Project()        
        
        self.ui.textBrowser_log.append(WELCOME_MESSAGE)
        print(WELCOME_MESSAGE)
        
        subwindow = SubWindowHtmlView()
        subwindow.display_htmlfile(os.path.join(os.path.dirname(__file__), 'README.html'))
        self.ui.mdiArea.addSubWindow(subwindow)
        subwindow.show()
        
        
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
        
    def link_gui_item_to_variable(self, gui_item: QLineEdit | QCheckBox, var_name: str, type_func) -> None:        
        def change_value(value):
            self.active_model.__setattr__(var_name, type_func(value))
        
        if isinstance(gui_item, QLineEdit):
            gui_item.setText(str(self.active_model.__getattribute__(var_name)))
            gui_item.textChanged.connect(change_value)
        else:
            gui_item.setChecked(bool(self.active_model.__getattribute__(var_name)))
            gui_item.clicked.connect(change_value)
            
    def set_progressbar(self, state: Literal['begin', 'end']) -> None:
        if state == 'begin':
            self.ui.progressBar.setMinimum(0)
            self.ui.progressBar.setMaximum(0)
        else:
            self.ui.progressBar.setMinimum(0)
            self.ui.progressBar.setMaximum(100)
            self.ui.progressBar.setValue(100)
        
    def import_parts(self) -> None:
        filename_list, _ = QFileDialog.getOpenFileNames(self, caption='Select Model File(s)', dir='./', filter="All Files (*);;stl Files (*.stl);;math model Files (*.py)")
        for filename in filename_list:
            self.project.import_part(filename)
            self.gui_log('info', f'imported {filename}')
        self.refresh_qitemmodel_models()
        
    def add_to_assembly(self) -> None:
        assembly_key = self.ui.comboBox_assemblies.currentText()
        self.project.add_part_to_assembly(self.active_model.key, assembly_key)
        self.refresh_qitemmodel_models()
        
    def new_assembly(self) -> None:
        self.project.new_assembly()
        self.refresh_qitemmodel_models()
        self.refresh_combobox_assemblies()
        
    def user_defined_model(self):
        self.subwindow_userdefined_model = SubWindowUserDefinedModel(self)
        self.subwindow_userdefined_model.show()
        
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
        # self.ui.pushButton_sample.setDisabled(False)
        self.ui.pushButton_sample.setText('Sample')
        self.ui.pushButton_add_transform.setDisabled(False)
        self.ui.pushButton_delete_selected_transform.setDisabled(False)
        self.ui.pushButton_apply_transform.setDisabled(False)
        # self.ui.pushButton_plot_model.setDisabled(False)
        # self.ui.pushButton_scatter.setDisabled(False)
        self.ui.pushButton_scatter.setText('Virtual Scatter')
        self.refresh_qitemmodel_params()
        self.refresh_qitemmodel_transforms()
        
        
    def _unique_actions_for_assembly_model_selected(self) -> None:
        self.ui.pushButton_add_to_assembly.setDisabled(True)
        self.ui.tableView_model_params.setDisabled(True)
        # self.ui.pushButton_sample.setDisabled(False)
        self.ui.pushButton_sample.setText('Sample All Sub-Parts')
        self.ui.pushButton_add_transform.setDisabled(True)
        self.ui.pushButton_delete_selected_transform.setDisabled(True)
        self.ui.pushButton_apply_transform.setDisabled(True)
        # self.ui.pushButton_plot_model.setDisabled(False)
        # self.ui.pushButton_scatter.setDisabled(False)
        self.ui.pushButton_scatter.setText('Virtual Scatter by All Sub-Parts')
        self.qitemmodel_params.clear()
        self.qitemmodel_transforms.clear()
    
    def _actions_for_model_selected(self) -> None:
        self.ui.label_active_model.setText(
            f'Active Model: 【{self.active_model.key}】 {self.active_model.name}'
        )
        self.link_gui_item_to_variable(self.ui.lineEdit_real_lattice_1d_size, 'real_lattice_1d_size', int)
        self.link_gui_item_to_variable(self.ui.lineEdit_q1d_min, 'q1d_min', float)
        self.link_gui_item_to_variable(self.ui.lineEdit_q1d_max, 'q1d_max', float)
        self.link_gui_item_to_variable(self.ui.lineEdit_q1d_num, 'q1d_num', int)
        self.link_gui_item_to_variable(self.ui.checkBox_q1d_log_spaced, 'q1d_log_spaced', bool)
        self.link_gui_item_to_variable(self.ui.lineEdit_det_res_h, 'det_res_h', int)
        self.link_gui_item_to_variable(self.ui.lineEdit_det_res_v, 'det_res_v', int)
        self.link_gui_item_to_variable(self.ui.lineEdit_det_pixel_size, 'det_pixel_size', float)
        self.link_gui_item_to_variable(self.ui.lineEdit_wavelength, 'det_wavelength', float)
        self.link_gui_item_to_variable(self.ui.lineEdit_det_sdd, 'det_sdd', float)
        self.link_gui_item_to_variable(self.ui.checkBox_log_Idet, 'log_Idet', bool)
        self.link_gui_item_to_variable(self.ui.lineEdit_q3d_qmax, 'q3d_max', float)
        self.link_gui_item_to_variable(self.ui.checkBox_log_I3d, 'log_I3d', bool)

        
    def refresh_qitemmodel_params(self) -> None:
        self.qitemmodel_params.clear()
        self.qitemmodel_params.setHorizontalHeaderLabels(['Param', 'Value'])
        if isinstance(self.active_model, StlPartWrapper):
            params = dict(sld_value=self.active_model.sld_value)
        elif isinstance(self.active_model, MathPartWrapper):
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
        if isinstance(self.active_model, StlPartWrapper):
            self.active_model.sld_value = \
                float(self.qitemmodel_params.index(0, 1).data())
            self.gui_log('success', 'change parameter')
        elif isinstance(self.active_model, MathPartWrapper):
            for i in range(self.qitemmodel_params.rowCount()):
                param_name = self.qitemmodel_params.index(i, 0).data()
                param_value = float(self.qitemmodel_params.index(i, 1).data())
                self.active_model.set_params(**{param_name: param_value})
            self.gui_log('success', 'change parameter')
            
    def refresh_qitemmodel_transforms(self) -> None:
        self.qitemmodel_transforms.clear()
        self.qitemmodel_transforms.setHorizontalHeaderLabels(['Type', 'Vector/Axis', 'Angle (deg)'])
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
        self.set_progressbar('begin')
                
    def sample_thread_end(self) -> None:
        self.gui_log('success', 'Sample done')
        self.set_progressbar('end')
        
    def plot_model(self) -> None:
        if self.ui.radioButton_volume_plot.isChecked():
            plot_type = 'volume'
        else: # self.ui.radioButton_voxel_plot.isChecked()
            plot_type = 'voxel'
        subwindow = SubWindowPlotView(self)
        subwindow.setWindowTitle(f'【{self.active_model.key}】Model Plot')
        subwindow.plot_model(self.active_model, plot_type)
        self.ui.mdiArea.addSubWindow(subwindow)
        subwindow.show()
        self.gui_log('success', 'Plot done')
        self.set_progressbar('end')
        
    def scatter(self) -> None:
        self.thread = GeneralThread(
            self.active_model.scatter
        )
        self.thread.thread_end.connect(self.scatter_thread_end)
        self.thread.start()
        self.set_progressbar('begin')        
                
    def scatter_thread_end(self) -> None:
        self.gui_log('success', 'Scatter done')
        self.set_progressbar('end')
        
    def measure_1d(self) -> None:
        q1d = self.active_model.gen_q('1d')
        self.thread = MeasureThread(
            self.active_model.measure,
            q1d
        )
        self.thread.thread_end.connect(self.display_sas_plot_subwindow)
        self.thread.start()
        self.set_progressbar('begin')
        
    def measure_det(self) -> None:
        qx, qy, qz = self.active_model.gen_q('det')
        self.thread = MeasureThread(
            self.active_model.measure,
            qx, qy, qz
        )
        self.thread.thread_end.connect(self.display_sas_plot_subwindow)
        self.thread.start()
        self.set_progressbar('begin')
    
    def measure_3d(self) -> None:
        qx, qy, qz = self.active_model.gen_q('3d')
        self.thread = MeasureThread(
            self.active_model.measure,
            qx, qy, qz
        )
        self.thread.thread_end.connect(self.display_sas_plot_subwindow)
        self.thread.start()
        self.set_progressbar('begin')
        
    def display_sas_plot_subwindow(self, data: tuple[Tensor, ...]) -> None:
        self.gui_log('success', 'Measure done')
        subwindow = SubWindowPlotView(self)
        subwindow.plot_sas(*data)
        self.ui.mdiArea.addSubWindow(subwindow)
        subwindow.show()
        self.gui_log('success', 'Plot done')
        self.set_progressbar('end')
    






        
        
        
def run():
    app = QApplication(sys.argv)
    main_window = MainWindow()
    # main_window = SubWindowScatterPlot()
    sys.exit(app.exec())

if __name__ == '__main__':
    run()