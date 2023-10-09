import sys

import pytest
from PySide6.QtWidgets import QApplication

from model2sas.gui.main import run


def test_gui():
    with pytest.raises(SystemExit) as e:
        run()
    assert e.type == SystemExit
    assert e.value.code == 0

if __name__ == '__main__':
    test_gui()