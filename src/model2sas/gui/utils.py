import os
import tempfile
import time
import hashlib
from typing import Callable

from PySide6.QtCore import QThread, Signal


def time_hash_digest(length: int) -> str:
    return hashlib.sha1(
            str(time.time()).encode('utf-8')
            ).hexdigest()[:length]


class GeneralThread(QThread):
    thread_end = Signal()
    def __init__(self, func: Callable, *args, **kwargs) -> None:
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
        I = self.func(*self.args, **self.kwargs)
        result = tuple([*self.args, I])  # return q and I
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