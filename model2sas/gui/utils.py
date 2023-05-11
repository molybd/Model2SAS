import os

from model2sas.model import Assembly, Part

from ..model import Part, StlPart, MathPart, Assembly


class ModelContainer:
    """Container for part model or assembly model.
    Record params for GUI
    """    
    def __init__(self) -> None:
        self.q1d_min: float = 0.01
        self.q1d_max: float = 1.0
        self.q1d_num: int = 200
        self.q1d_log_spaced: bool = False
        self.real_lattice_1d_size: int = 50
        
        
class PartContainer(ModelContainer):
    """_summary_
    """
    def __init__(self, key: str, model: Part) -> None:
        super().__init__()
        self.key = key
        self.model = model
        self.type = 'part'
        
        
class AssemblyContainer(ModelContainer):
    """_summary_
    """
    def __init__(self, key: str, model: Assembly) -> None:
        super().__init__()
        self.key = key
        self.model = model
        self.type = 'assembly'


class Project:
    """Container for GUI project
    """    
    def __init__(self) -> None:
        self.devices: list = ['cpu']
        self.unit: str = 'angstrom'
        self.parts: dict[str, PartContainer] = {}
        self.assembly: AssemblyContainer
        
    def new_assembly(self):
        self.assembly = AssemblyContainer('assembly', Assembly())
        
    def add_part(self, filename: str, device: str):
        stripped_name = os.path.splitext(os.path.basename(filename))[0]
        i = 0
        while True:
            partname = '{}_{}'.format(stripped_name, i)
            if partname not in self.parts.keys():
                break
            else:
                i += 1
        if os.path.splitext(filename)[-1].lower() == '.stl':
            part = StlPart(filename=filename, partname=partname, device=device)
        elif os.path.splitext(filename)[-1].lower() == '.py':
            part = MathPart(filename=filename, partname=partname, device=device)
        else:
            raise TypeError()
        self.parts[partname] = PartContainer(partname, part)
        self.assembly.model.add_part(part)
        