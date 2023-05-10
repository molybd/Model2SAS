import sys
from typing import Optional
import PySide6.QtCore

from PySide6.QtWidgets import QApplication, QMainWindow, QWidget, QFileDialog

from .MainWindow_ui import Ui_MainWindow
from .SubWindow_buildmath_ui import Ui_subWindow_build_math_model

# from .. import model2sas
# from model2sas import StlPart, MathPart, Assembly


class MainWindow(QMainWindow):
    
    def __init__(self) -> None:
        # ui related
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.show()
        
        # data related
        self.model2sas_models = dict()
        
    def load_model_files(self):
        print('clicked load_model_files')
        filename_list, _ = QFileDialog.getOpenFileNames(self, caption='Select Model File(s)', dir='./', filter="All Files (*);;stl Files (*.stl);;math model Files (*.py)")
        print(filename_list)
        # for filename in filename_list:
        #     model = 
        
        
    def build_math_model(self):
        print('clicked build_math_model')
        subwindow_build_math_model = SubWindowBuildMathModel()
        self.ui.mdiArea.addSubWindow(subwindow_build_math_model)
        subwindow_build_math_model.show()
        
        
class SubWindowBuildMathModel(QWidget):
    
    def __init__(self) -> None:
        super().__init__()
        self.ui = Ui_subWindow_build_math_model()
        self.ui.setupUi(self)
        
        
if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = MainWindow()
    sys.exit(app.exec())
