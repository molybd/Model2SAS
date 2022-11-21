'''Standard sample model mainly for test
and evaluation use.
'''

import torch
from torch import Tensor


class Standard:
    '''Parent model for different standard samples.
    Inherited from ..model.Model class, have get_F_value
    and get_s_max 2 methods, can be directly used as a
    model instead of import as StlPart or MathPart.
    Initialized by d-spacings.
    '''
    def __init__(self, d_spacing_list: list, device: str = 'cpu') -> None:
        self.device = device
        self.d_spacing_list = d_spacing_list

    def get_F_value(self, reciprocal_coord: Tensor) -> Tensor:
        s = torch.sqrt(torch.sum(reciprocal_coord**2, dim=1))
        F = torch.ones_like(s, dtype=torch.complex64)
        for i, d_spacing in enumerate(self.d_spacing_list):
            peak_position = 1 / d_spacing
            sigma = 0.015*peak_position
            height = 1e3 / (i+1)
            F = F + height * torch.exp( -(s-peak_position)**2 / (2*sigma**2) )
        return F

    def get_s_max(self) -> float:
        return 1 / min(self.d_spacing_list)

# unit of d-spacing is Angstrom here
AgBh = Standard([58.378/(i+1) for i in range(10)])
bull_tendon = Standard([653/(i+1) for i in range(10)])