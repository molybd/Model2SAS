import os
from typing import Literal

from ..model import Part, StlPart, MathPart, Assembly


class ModelContainer:
    """Container for part model or assembly model.
    Record params for GUI
    """    
    def __init__(self, key: str, model: StlPart | MathPart | Part | Assembly) -> None:
        self.key = key
        self.model = model
        self.type: Literal['part', 'assembly']
        if isinstance(model, Assembly):
            self.type = 'assembly'
        else:
            self.type = 'part'
        self.q1d_min: float = 0.01
        self.q1d_max: float = 1.0
        self.q1d_num: int = 200
        self.q1d_log_spaced: bool = False
        self.real_lattice_1d_size: int = 50
        
        if self.type == 'assembly':
            self.children: list[str] = []
        else:
            self.parent: list[str] = []


class Project:
    """Container for GUI project
    """    
    def __init__(self) -> None:
        self.devices: list = ['cpu']
        self.unit: str = 'angstrom'
        self.parts: dict[str, ModelContainer] = {}
        self.assemblies: dict[str, ModelContainer] = {}
        self.name_index: dict[str, int] = {}
        
    def new_assembly(self, name: str):
        if name in self.name_index.keys():
            self.name_index[name] += 1
        else:
            self.name_index[name] = 0
        key = '{}_{}'.format(name, self.name_index[name])
        self.assemblies[key] = ModelContainer(key, Assembly())
        
    def load_part_from_file(self, filename: str):
        name = os.path.splitext(os.path.basename(filename))[0]
        if name in self.name_index.keys():
            self.name_index[name] += 1
        else:
            self.name_index[name] = 0
        key = '{}_{}'.format(name, self.name_index[name])
        
        if os.path.splitext(filename)[-1].lower() == '.stl':
            part = StlPart(filename=filename, partname=key)
        elif os.path.splitext(filename)[-1].lower() == '.py':
            part = MathPart(filename=filename, partname=key)
        else:
            raise TypeError()
        self.parts[key] = ModelContainer(key, part)
        
    def add_part_to_assembly(self, part_key: str, assembly_key: str):
        self.assemblies[assembly_key].children.append(part_key)
        self.parts[part_key].parent.append(assembly_key)
        