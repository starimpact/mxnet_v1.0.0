KVStore Speical for Customized KVStore
=====
Special KVStore is used for some cases like model parallel for full connection layer.

Support concat, sum, average for now.
You can define your style operation in the python side easily.

Interface:
----------
* set_kvspecialer
* set_optimizer
* init_kvspecial
* push_kvspecial
* pull_kvspecial

Demo:
----------
More detail is in the folder test_kvstore
```python
from mxnet import kvspecial

keys = ['3','5','9']  #support string type keys.
concat_keys = ['10000001', '10000002']
shape = (512,5120)
concat_stype = 'alone_concat' # 'alone' means only calculate in one server. 'concat' means do concat operation.
kv = mx.kv.create('dist_device_sync') #only sync mode is supported.
kv.set_kvspecialer(kvspecial.KVSpecial())
kv.set_optimizer(mx.optimizer.create('test', rescale_grad=0.1))
kv.init_kvspecial(concat_keys[0], mx.nd.zeros(shape), concat_stype)
my_rank = kv.rank
def go_concat(stype):
    rnd = mx.nd.ones(shape) + my_rank
    kv.push_kvspecial(concat_keys[0], rnd, stype)
    if 'concat' in stype:
        val = mx.nd.zeros((shape[0]*2, shape[1]))
    kv.pull_kvspecial(concat_keys[0], val, stype)
    print 'val', val[0:2,0:2], 'val end', val[-3:,-3:], kv.rank

go_concat(concat_stype)
```

Customized Operation Interface
----------
In the python file python/mxnet/kvspecial.py
```python
 18 class KVSpecial(object):
 19   def __init__(self):
 20     pass
 21
 22   def __call__(self, key, inlist, out, kvtype):
 23     if 'concat' in kvtype:
 24       self.concat(key, inlist, out)
 25     elif 'sum' in kvtype:
 26       self.sum_(key, inlist, out)
 27     elif 'max' in kvtype:
 28       self.max_(key, inlist, out)
 29     else:
 30       print 'unknown kvtype:', kvtype
 31     pass
 32
 33   def concat(self, key, inlist, out):
 34     shape_0 = inlist[0].shape
 35     in_num = len(inlist)
 36     shape_out = out.shape
 37     assert in_num*shape_0[0]==shape_out[0]
 38     assert shape_0[1]==shape_out[1]
 39     for i in xrange(in_num):
 40       out[i*shape_0[0]:(i+1)*shape_0[0]] = inlist[i]
 41     pass
 42
 43   def sum_(self, key, inlist, out):
 44     shape_0 = inlist[0].shape
 45     in_num = len(inlist)
 46     shape_out = out.shape
 47     assert shape_0==shape_out
 48     assert shape_0[1]==shape_out[1]
 49     out[:] = inlist[0]
 50     for i in xrange(1, in_num):
 51       out += inlist[i]
 52     pass
 53
 54   def max_(self, key, inlist, out):
 55     shape_0 = inlist[0].shape
 56     in_num = len(inlist)
 57     shape_out = out.shape
 58     assert shape_0==shape_out
 59     assert shape_0[1]==shape_out[1]
 60     out[:] = inlist[0]
 61     for i in xrange(1, in_num):
 62       out[:] = maximum(out, inlist[i])
 63     pass
```

