"""A wrapper for convient using and recording operations in GUI.
Avoid direct using Part or Assembly api.
"""
import os, time, hashlib, copy, math
from typing import Literal

import torch
from torch import Tensor

from ..model import Part, StlPart, MathPart, Assembly


class Model:
    """_summary_
    """    
    def __init__(self, model: StlPart | MathPart | Part | Assembly) -> None:
        self.key = self._gen_key()
        # self.model = model
        self.type: Literal['stlpart', 'mathpart', 'assembly', 'generalpart']
        if isinstance(model, Assembly):
            self.type = 'assembly'
        elif isinstance(model, StlPart):
            self.type = 'stlpart'
        elif isinstance(model, MathPart):
            self.type = 'mathpart'
        else:
            self.type = 'generalpart'
        self.q1d_min: float = 0.01
        self.q1d_max: float = 1.0
        self.q1d_num: int = 200
        self.q1d_log_spaced: bool = False
        self.real_lattice_1d_size: int = 50
        
    def _gen_key(self) -> str:
        return hashlib.sha1(
            str(time.time()).encode('utf-8')
            ).hexdigest()[:4]
        
    def gen_q1d(self) -> Tensor:
        if self.q1d_log_spaced:
            q1d = torch.logspace(
                math.log10(self.q1d_min),
                math.log10(self.q1d_max),
                self.q1d_num
            )
        else:
            q1d = torch.linspace(
                self.q1d_min,
                self.q1d_max,
                self.q1d_num
            )
        return q1d
        

class PartModel(Model):
    """_summary_
    """    
    def __init__(self, model: StlPart | MathPart | Part) -> None:
        super().__init__(model)
        self.model = model
        self.name = model.partname or 'no_name'
        self.geo_transform = copy.deepcopy(self.model.geo_transform)
        
    def sample(self) -> None:
        self.model.sample()
        
    def scatter(self) -> None:
        self.model.scatter()
        
    def measure(self) -> tuple[Tensor, Tensor]:
        q1d = self.gen_q1d()
        return q1d, self.model.measure(q1d)
        
        
class StlPartModel(PartModel):
    """_summary_
    """    
    def __init__(self, model: StlPart) -> None:
        super().__init__(model)
        self.model = model
        self.sld_value = model.sld_value
        
    def sample(self) -> None:
        self.model.sld_value = self.sld_value
        self.model.sample(real_lattice_1d_size=self.real_lattice_1d_size)
        

class MathPartModel(PartModel):
    """_summary_
    """    
    def __init__(self, model: MathPart) -> None:
        super().__init__(model)
        self.model = model
        
    def get_params(self) -> dict:
        return self.model.get_params()
    
    def set_params(self, **kwargs) -> None:
        self.model.set_params(**kwargs)
        
    def sample(self) -> None:
        self.model.sample(real_lattice_1d_size=self.real_lattice_1d_size)
        
        

class AssemblyModel(Model):
    """_summary_
    """    
    def __init__(self, model: Assembly) -> None:
        super().__init__(model)
        self.name = 'assembly'
        self.model = model
        self.parts: list[StlPartModel| MathPartModel | PartModel] = []
        
    def _rebuild_parts_in_assembly(self) -> None:
        self.model.parts.clear()
        self.model.parts = [partmodel.model for partmodel in self.parts]
    
    def sample(self) -> None:
        """sample all sub-parts
        """        
        self._rebuild_parts_in_assembly()
        for model in self.parts:
            model.sample()
            
    def scatter(self) -> None:
        """scatter all sub-parts
        """        
        self._rebuild_parts_in_assembly()
        for model in self.parts:
            model.scatter()
            
    def measure(self) -> tuple[Tensor, Tensor]:       
        q1d = self.gen_q1d()
        self._rebuild_parts_in_assembly()
        return q1d, self.model.measure(q1d)


class Project:
    """_summary_
    """    
    def __init__(self) -> None:
        self.parts: dict[str, StlPartModel | MathPartModel | PartModel] = {}
        self.assemblies: dict[str, AssemblyModel] = {}
        
    def import_part(self, filename: str) -> None:
        if os.path.splitext(filename)[-1].lower() == '.stl':
            part = StlPartModel(StlPart(filename=filename))
        elif os.path.splitext(filename)[-1].lower() == '.py':
            part = MathPartModel(MathPart(filename=filename))
        else:
            raise TypeError()
        self.parts[part.key] = part
        
    def new_assembly(self) -> None:
        assmbly = AssemblyModel(Assembly())
        self.assemblies[assmbly.key] = assmbly
        
    def add_part_to_assembly(self, part: str | StlPartModel | MathPartModel | PartModel, assembly: str | AssemblyModel) -> None:
        if isinstance(part, str):
            part = self.parts[part]
        if isinstance(assembly, str):
            assembly = self.assemblies[assembly]
        assembly.parts.append(part)
    

    