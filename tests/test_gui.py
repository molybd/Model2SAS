import sys

import pytest
from PySide6.QtWidgets import QApplication

from model2sas.gui.main import MainWindow


def test_gui():
    with pytest.raises(SystemExit) as e:
        app = QApplication(sys.argv)
        main_window = MainWindow()
        sys.exit(app.exec())
    assert e.type == SystemExit
    assert e.value.code == 0
