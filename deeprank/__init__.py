# h5py complains since the last numpy update ...
# the warning is
# /home/nico/anaconda3/lib/python3.6/site-packages/h5py/__init__.py:36: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.
#  from ._conv import register_converters as _register_converters
import logging.config
import logging
from .tools import *
from .generate import *
from . import global_settings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# deep learning
# import torch fals on Travis
#from .learn import *

logging.config.dictConfig(global_settings.DEFAULT_LOGGING)
logger = logging.getLogger(__name__)

print(f'"name" is {__name__}')
