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
    if 'concat' in kvtype:
      concat(key, inlist, out)
    elif 'sum' in kvtype:
      sum(key, inlist, out)
    elif 'max' in kvtype:
      max(key, inlist, out)
    else:
      print 'unknown kvtype:', kvtype
    pass

  def concat(self, key, inlist, out):
    shape_0 = inlist[0].shape
    in_num = len(inlist)
    shape_out = out.shape
    assert in_num*shape_0[0]==shape_out[0]
    assert shape_0[1]==shape_out[1]
    for i in xrange(in_num):
      out[i*shape_0[0]:(i+1)*shape_0[0]] = inlist[i]
    pass

  def sum(self, key, inlist, out):
    shape_0 = inlist[0].shape
    in_num = len(inlist)
    shape_out = out.shape
    assert shape_0==shape_out
    assert shape_0[1]==shape_out[1]
    out[:] = inlist[0]
    for i in xrange(1, in_num):
      out += inlist[i]
    pass

  def max(self, key, inlist, out):
    shape_0 = inlist[0].shape
    in_num = len(inlist)
    shape_out = out.shape
    assert shape_0==shape_out
    assert shape_0[1]==shape_out[1]
    out[:] = inlist[0]
    for i in xrange(1, in_num):
      out[:] = mx.nd.maximum(out, inlist[1])
    pass
