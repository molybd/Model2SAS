import sys
from typing import Optional
import PySide6.QtCore

from PySide6.QtWidgets import QApplication, QMainWindow, QWidget, QFileDialog

from MainWindow_ui import Ui_MainWindow
from SubWindow_buildmath_ui import Ui_subWindow_build_math_model


class MainWindow(QMainWindow):
    
    def __init__(self) -> None:
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.show()
        
    def browse_model_files(self):
        print('clicked browse_model_files')
        filename_list, filetype_list = QFileDialog.getOpenFileNames(self, caption='Select Model File(s)', dir='./', filter="All Files (*);;stl Files (*.stl);;math model Files (*.py)")
        print(filename_list)
        
        
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
