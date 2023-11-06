import tempfile
import os
from typing import Callable, Literal, Optional
import shutil

from PySide6.QtCore import QThread, Signal, Slot, QObject, QModelIndex, SIGNAL, SLOT, QSize
from PySide6.QtWidgets import QApplication, QMainWindow, QWidget, QFileDialog, QInputDialog, QMdiSubWindow, QHeaderView, QStyledItemDelegate, QComboBox, QLineEdit, QCheckBox, QGridLayout, QPushButton, QSpacerItem, QSizePolicy, QLabel
from PySide6.QtWebEngineWidgets import QWebEngineView
import plotly.graph_objects as go
import torch
from torch import Tensor
import numpy as np

from .utils import GeneralThread, PlotThread, time_hash_digest
from ..plot import write_html, plot_1d_sas, plot_2d_sas, plot_3d_sas, plot_model
from .model_wrapper import ModelWrapperType


class SubWindowHtmlView(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setup_ui()
        self.resize(500, 400)  
        self.setMinimumSize(QSize(400, 400)) # begin size can't be too small or plot will be blank
        # self.show()
        
    def setup_ui(self) -> None:
        self.gridLayout = QGridLayout(self)
        self.webEngineView = QWebEngineView(self)
        self.gridLayout.addWidget(self.webEngineView, 0, 0, 1, 1)
        
    def display_htmlfile(self, htmlfile: str):
        self.webEngineView.setUrl(htmlfile.replace('\\', '/')) #must use forward slashes in file path, or will be blank or error
        
        
class SubWindowPlotView(SubWindowHtmlView):
    
    def __init__(self, mainwindow) -> None:
        super().__init__()
        self.mainwindow = mainwindow
        self.window_type: Literal['1d-sas', 'det-sas', '3d-sas', 'model']
        self.thread: QThread
        self.fig: go.Figure
        self.htmlfile = os.path.join(
            tempfile.gettempdir(), f'model2sas_plot_{time_hash_digest(4)}.html'
            )
        self.save_image_params: dict[Literal['width', 'height',  'scale'], int | float] = dict(width=800, height=600, scale=2)
    
    def setup_ui(self) -> None:
        self.gridLayout = QGridLayout(self)
        self.webEngineView = QWebEngineView(self)
        self.gridLayout.addWidget(self.webEngineView, 0, 0, 1, 3)
        self.horizontalSpacer = QSpacerItem(311, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)
        self.pushButton_save_data = QPushButton(self)
        self.pushButton_save_data.setText('Save Data')
        self.pushButton_save_data.clicked.connect(self.save_data)
        self.pushButton_save_image = QPushButton(self)
        self.pushButton_save_image.setText('Save Image')
        self.pushButton_save_image.clicked.connect(self.save_image)
        
        self.gridLayout.addItem(self.horizontalSpacer, 1, 0, 1, 1)
        self.gridLayout.addWidget(self.pushButton_save_data, 1, 1, 1, 1)
        self.gridLayout.addWidget(self.pushButton_save_image, 1, 2, 1, 1)
        
        self.pushButton_save_data.setDisabled(True)
        self.pushButton_save_image.setDisabled(True)
        
    def plot_sas(self, *data: Tensor) -> None:
        self.pushButton_save_image.setEnabled(True)
        self.pushButton_save_data.setEnabled(True)
        if len(data) == 2: # 1d sas curve plot
            self.window_type = '1d-sas'
            self.setWindowTitle(f'【{self.mainwindow.active_model.key}】1-D Scatter Plot')
            self.q, self.I = data
            self.fig = plot_1d_sas(
                self.q,
                self.I,
                mode = 'lines',
                name = f'【{self.mainwindow.active_model.key}】 {self.mainwindow.active_model.name}',
                title = f'【{self.mainwindow.active_model.key}】 {self.mainwindow.active_model.name}',
                savename = self.htmlfile,
                show = False
            )
        elif len(data) == 4 and data[-1].dim() == 2: # 2d detector plot
            self.window_type = 'det-sas'
            self.setWindowTitle(f'【{self.mainwindow.active_model.key}】Detector Scatter Plot')
            self.qx, self.qy, self.qz, self.I2d = data
            self.fig = plot_2d_sas(
                self.I2d,
                logI = self.mainwindow.active_model.log_Idet,
                title = f'【{self.mainwindow.active_model.key}】 {self.mainwindow.active_model.name}',
                savename = self.htmlfile,
                show = False
            )
        elif len(data) == 4 and data[-1].dim() == 3: # 3d sas plot
            self.window_type = '3d-sas'
            self.setWindowTitle(f'【{self.mainwindow.active_model.key}】3-D Scatter Plot')
            self.qx, self.qy, self.qz, self.I3d = data
            self.fig = plot_3d_sas(
                self.qx, self.qy, self.qz, self.I3d,
                logI = self.mainwindow.active_model.log_I3d,
                title = f'【{self.mainwindow.active_model.key}】 {self.mainwindow.active_model.name}',
                savename = self.htmlfile,
                show = False
            )
        self.webEngineView.setUrl(self.htmlfile.replace('\\', '/')) #must use forward slashes in file path, or will be blank or error
            
    def plot_model(self, model_wrapper: ModelWrapperType, plot_type: Literal['volume', 'voxel']) -> None:
        self.window_type = 'model'
        self.pushButton_save_image.setEnabled(True)
        self.setWindowTitle(f'【{model_wrapper.key}】Model Plot')
        if model_wrapper.type == 'assembly':
            model = model_wrapper.model.parts
        else:
            model = [model_wrapper.model]
        self.fig = plot_model(
            *model,
            type = plot_type,
            title = f'Model Plot: {model_wrapper.key}',
            savename = self.htmlfile,
            show = False
        )
        self.webEngineView.setUrl(self.htmlfile.replace('\\', '/')) #must use forward slashes in file path, or will be blank or error
        
    def save_image(self) -> None:
        self.save_image_window = SubSubWindowSaveImage(self)
        self.save_image_window.show()
    
    def save_data(self) -> None:
        if self.window_type == '1d-sas':
            filename, _ = QFileDialog.getSaveFileName(self, caption='Save 1-D Data', dir='./', filter="DAT (*.dat)")
            data = torch.stack((self.q, self.I), dim=1).numpy()
            np.savetxt(filename, data, delimiter='\t', header='Simulated by Model2SAS\nq\tI')
        elif self.window_type == 'det-sas':
            filename, _ = QFileDialog.getSaveFileName(self, caption='Save Detector Data', dir='./', filter="NPZ (*.npz)")
            np.savez_compressed(filename, qx=self.qx.numpy(), qy=self.qy.numpy(), qz=self.qz.numpy(), I2d=self.I2d.numpy())
        elif self.window_type == '3d-sas':
            filename, _ = QFileDialog.getSaveFileName(self, caption='Save 3-D Data', dir='./', filter="NPZ (*.npz)")
            np.savez_compressed(filename, qx=self.qx.numpy(), qy=self.qy.numpy(), qz=self.qz.numpy(), I3d=self.I3d.numpy())
            
    def __del__(self):
        '''Remove temp file when destroyed
        '''
        os.remove(self.htmlfile)
        
        
class SubSubWindowSaveImage(QWidget):
    def __init__(self, parent_window: SubWindowPlotView) -> None:
        super().__init__()
        self.parent_window = parent_window
        self.setup_ui()
        
    def setup_ui(self) -> None:
        self.setWindowTitle('Save Image')
        self.gridLayout = QGridLayout(self)
        self.label_width = QLabel(self)
        self.label_width.setText('Width')
        self.gridLayout.addWidget(self.label_width, 0, 0, 1, 1)
        self.lineedit_width = QLineEdit(self)
        self.lineedit_width.setText(str(self.parent_window.save_image_params['width']))
        self.gridLayout.addWidget(self.lineedit_width, 0, 1, 1, 1)
        self.label_height = QLabel(self)
        self.label_height.setText('Height')
        self.gridLayout.addWidget(self.label_height, 1, 0, 1, 1)
        self.lineedit_height = QLineEdit(self)
        self.lineedit_height.setText(str(self.parent_window.save_image_params['height']))
        self.gridLayout.addWidget(self.lineedit_height, 1, 1, 1, 1)
        self.label_scale = QLabel(self)
        self.label_scale.setText('Scale')
        self.gridLayout.addWidget(self.label_scale, 2, 0, 1, 1)
        self.lineedit_scale = QLineEdit(self)
        self.lineedit_scale.setText(str(self.parent_window.save_image_params['scale']))
        self.gridLayout.addWidget(self.lineedit_scale, 2, 1, 1, 1)
        self.pushbutton_save_image = QPushButton(self)
        self.pushbutton_save_image.setText('Save Image')
        self.pushbutton_save_image.clicked.connect(self.save_image)
        self.gridLayout.addWidget(self.pushbutton_save_image, 3, 1, 1, 1)
        
    def save_image(self) -> None:
        width = int(self.lineedit_width.text())
        height = int(self.lineedit_height.text())
        scale = float(self.lineedit_scale.text())
        self.parent_window.save_image_params.update(
            width=width, height=height, scale=scale
        )
        filename, _ = QFileDialog.getSaveFileName(self, caption='Save Image', dir='./', filter="PNG (*.png);;JPEG (*.jpg);;SVG (*.svg);;PDF (*.pdf);;HTML (*.html);;All Files (*)")
        if os.path.splitext(filename)[-1] == '.html':
            shutil.copyfile(self.parent_window.htmlfile, filename)
        else:
            self.parent_window.fig.write_image(
                filename, width=width, height=height, scale=scale
            )
        self.close()