import os

from pydantic import BaseModel

from options.options_enum import AutodifEnum, SolverEnum
from time_series.symplectic_euler import add_symplectic_euler

dirname = os.path.dirname(__file__)


class OdeintOptions(BaseModel):

    autodif: AutodifEnum = AutodifEnum.naive
    atol: float = 1.0e-5
    rtol: float = 1e-8
    step_size: float = 0.05
    solver: SolverEnum = SolverEnum.rk4
    if solver == SolverEnum.symplectic_euler and autodif == AutodifEnum.adjoint:
        raise ValueError(f"Options solver: {solver} and {autodif} cannot be combined")
    if solver == SolverEnum.symplectic_euler:
        add_symplectic_euler()

    class Config:
        use_enum_values = True
