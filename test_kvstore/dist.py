#!/usr/bin/env python

# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

# pylint: skip-file
import sys
sys.path.insert(0, "/home/mingzhang/work/dmlc/python_mxnet/python")
import mxnet as mx
import numpy as np
import numpy.random as rnd
from mxnet import kvspecial


keys = ['3','5','9']
sum_keys = [100000+1,100000+2]
concat_keys = [1000000+1, 1000000+2]
shape = (400,400)
big_shape = (400, 400)
rate = 1

stype = 'reduce_sum_alone'
#stype = 'concat_alone'

kv = mx.kv.create('dist_sync')
kv.set_kvspecialer(kvspecial.KVSpecial())
my_rank = kv.rank


rnd = mx.nd.random.uniform(0,1,shape)

print 'my rank:', my_rank, 'rnd', rnd[0:2, 0:2], 'rnd end', rnd[-3:,-3:]


kv.init_kvspecial(sum_keys[0], rnd, stype)
kv.push_kvspecial(sum_keys[0], rnd + my_rank, stype)

if 'concat' in stype:
    val = mx.nd.zeros((shape[0]*2, shape[1]))
else:
    val = mx.nd.zeros(shape)

kv.pull_kvspecial(sum_keys[0], val, stype)
print 'val', val[0:2,0:2], 'val end', val[-3:,-3:], kv.rank

if __name__ == "__main__":
    print 'haha', kv.rank
