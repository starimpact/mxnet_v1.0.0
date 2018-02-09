# coding: utf-8

import math
import pickle
import logging
import warnings
import numpy
from .base import py_str
from .ndarray import (NDArray, zeros, clip, sqrt, cast, maximum, abs as NDabs)
from .ndarray import (sgd_update, sgd_mom_update, adam_update, rmsprop_update, rmspropalex_update,
                      mp_sgd_update, mp_sgd_mom_update, square, ftrl_update)
from .ndarray import _internal
from .ndarray import op
from .ndarray import sparse
from .random import normal


class KVSpecial(object):
  def __init__(self):
    pass
  
  def __call__(self, key, inlist, out, kvtype):
    pass
