from .fno import TFNO, TFNO1d, TFNO2d, TFNO3d
from .fno import FNO, FNO1d, FNO2d, FNO3d
# only import SFNO if torch_harmonics is built locally
try:
    from .sfno import SFNO
except ModuleNotFoundError:
    pass
from .uno import UNO
from .transformer_no import TransformerNO
from .fnogno import FNOGNO
from .gino import GINO
from .base_model import get_model
