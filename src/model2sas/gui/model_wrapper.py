"""A wrapper for convient using and recording operations in GUI.
Avoid direct using Part or Assembly api.
"""
import os, time, hashlib, copy, math
from typing import Literal

import torch
from torch import Tensor

from ..model import Part, StlPart, MathPart, Assembly
from ..utils import Detector
from .utils import time_hash_digest


class ModelWrapper:
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
        
        self.model = model
        
        #! take care of the units!
        self.real_lattice_1d_size: int = 50
        self.q1d_min: float = 0.01  # reverse angstrom
        self.q1d_max: float = 1.0
        self.q1d_num: int = 200
        self.q1d_log_spaced: bool = False
        self.det_res_h: int = 981
        self.det_res_v: int = 1043
        self.det_pixel_size: float = 172  # micrometer
        self.det_wavelength: float = 1.54  # angstrom
        self.det_sdd: float = 2  # meter
        self.det_show_in_q_space: bool = False
        self.log_Idet: bool = True
        self.q3d_max: float = 0.5
        self.log_I3d: bool = True
        
        
    def _gen_key(self) -> str:
        return time_hash_digest(4)
    
    def gen_q(self, type: Literal['1d', 'det', '3d']) -> Tensor | tuple[Tensor, Tensor, Tensor]:
        match type:
            case 'det':
                det = Detector((self.det_res_h, self.det_res_v), self.det_pixel_size*1e-6)
                det.set_sdd(self.det_sdd)
                qx, qy, qz = det.get_reciprocal_coord(self.det_wavelength)
                return qx, qy, qz
            case '3d':
                q1d = torch.linspace(-self.q3d_max, self.q3d_max, steps=50)
                qx, qy, qz = torch.meshgrid(q1d, q1d, q1d, indexing='ij')
                return qx, qy, qz
            case _:
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
            
    def measure(self, *qi: Tensor, **kwargs) -> Tensor:
        return self.model.measure(*qi, **kwargs)
        

class PartWrapper(ModelWrapper):
    """_summary_
    """    
    def __init__(self, model: StlPart | MathPart | Part) -> None:
        super().__init__(model)
        self.name = model.partname or 'no_name'
        self.geo_transform = copy.deepcopy(self.model.geo_transform)
        
    def sample(self) -> None:
        self.model.sample()
        
    def scatter(self) -> None:
        self.model.scatter()
        
        
class StlPartWrapper(PartWrapper):
    """_summary_
    """    
    def __init__(self, model: StlPart) -> None:
        super().__init__(model)
        self.sld_value = model.sld_value
        
    def sample(self) -> None:
        self.model.sld_value = self.sld_value
        self.model.sample(real_lattice_1d_size=self.real_lattice_1d_size)
        

class MathPartWrapper(PartWrapper):
    """_summary_
    """    
    def get_params(self) -> dict:
        return self.model.get_params()
    
    def set_params(self, **kwargs) -> None:
        self.model.set_params(**kwargs)
        
    def sample(self) -> None:
        self.model.sample(real_lattice_1d_size=self.real_lattice_1d_size)
        
        

class AssemblyWrapper(ModelWrapper):
    """_summary_
    """    
    def __init__(self, model: Assembly) -> None:
        super().__init__(model)
        self.name = 'assembly'
        self.parts: list[StlPartWrapper| MathPartWrapper | PartWrapper] = []
        
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
            
    def measure(self, *qi: Tensor, **kwargs) -> Tensor:       
        self._rebuild_parts_in_assembly()
        return self.model.measure(*qi, **kwargs)


class Project:
    """_summary_
    """    
    def __init__(self) -> None:
        self.parts: dict[str, StlPartWrapper | MathPartWrapper | PartWrapper] = {}
        self.assemblies: dict[str, AssemblyWrapper] = {}
        
    def import_part(self, filename: str) -> None:
        if os.path.splitext(filename)[-1].lower() == '.stl':
            part = StlPartWrapper(StlPart(filename=filename))
        elif os.path.splitext(filename)[-1].lower() == '.py':
            part = MathPartWrapper(MathPart(filename=filename))
        else:
            raise TypeError()
        self.parts[part.key] = part
        
    def new_assembly(self) -> None:
        assmbly = AssemblyWrapper(Assembly())
        self.assemblies[assmbly.key] = assmbly
        
    def add_part_to_assembly(self, part: str | StlPartWrapper | MathPartWrapper | PartWrapper, assembly: str | AssemblyWrapper) -> None:
        if isinstance(part, str):
            part = self.parts[part]
        if isinstance(assembly, str):
            assembly = self.assemblies[assembly]
        assembly.parts.append(part)
    

ModelWrapperType = StlPartWrapper | MathPartWrapper | AssemblyWrapper | PartWrapper