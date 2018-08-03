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


#keys = [3,5,9]
#sum_keys = [100000+1,100000+2]
#concat_keys = [1000000+1, 1000000+2]
keys = ['3','5','9']
sum_keys = ['1000001','1000002']
concat_keys = ['10000001', '10000002']
#shape = (512,2048) #"failed"
#shape = (512,1024) #"ok"
shape = (512,5120) #"ok"
big_shape = (400, 400)
rate = 1

#stype = 'reduce_sum_alone'
concat_stype = 'alone_concat'

kv = mx.kv.create('dist_device_sync')
kv.set_kvspecialer(kvspecial.KVSpecial())
kv.set_optimizer(mx.optimizer.create('test', rescale_grad=0.1))

my_rank = kv.rank


if 1:
    sum_stype = 'reduce_sum'
    kv.init_kvspecial(sum_keys[0], mx.nd.zeros(shape), sum_stype)
kv.init_kvspecial(concat_keys[0], mx.nd.zeros(shape), concat_stype)


#def go_sum(stype):
#    rnd = mx.nd.random.uniform(0,1,shape) + my_rank
#    kv.push_kvspecial(sum_keys[0], rnd, stype)
#
#    print 'my rank:', my_rank, 'push top left value', rnd[0:2, 0:2], 'push bottom right value', rnd[-3:,-3:]
#    
#    if 'concat' in stype:
#        val = mx.nd.zeros((shape[0]*2, shape[1]))
#    else:
#        val = mx.nd.zeros(shape)
#    
#    #val_old = mx.nd.zeros(shape)
#    kv.pull_kvspecial(sum_keys[0], val, stype)
#    #kv.pull(keys[0], val_old)
#    #kv.push(keys[0], rnd)
#    #kv.pull(keys[0], val_old)
#    #print 'val', val[0:2,0:2], 'val end', val[-3:,-3:], kv.rank, 'val old', val_old[:2, :2], 'val old end', val_old[-3:, -3:]
#    print 'val', val[0:2,0:2], 'val end', val[-3:,-3:], kv.rank

def go_concat(stype):
    #rnd = mx.nd.random.uniform(0,1,shape) + my_rank
    rnd = mx.nd.ones(shape) + my_rank
    kv.push_kvspecial(concat_keys[0], rnd, stype)
    #print 'my rank:', my_rank, 'push top left value', rnd[0:2, 0:2], 'push bottom right value', rnd[-3:,-3:]
    
    if 'concat' in stype:
        val = mx.nd.zeros((shape[0]*2, shape[1]))
    else:
        val = mx.nd.zeros(shape)
    print 'go_concat', my_rank, rnd.shape, val.shape
    #val_old = mx.nd.zeros(shape)
    kv.pull_kvspecial(concat_keys[0], val, stype)
    #kv.pull(keys[0], val_old)
    #kv.push(keys[0], rnd)
    #kv.pull(keys[0], val_old)
    #print 'val', val[0:2,0:2], 'val end', val[-3:,-3:], kv.rank, 'val old', val_old[:2, :2], 'val old end', val_old[-3:, -3:]
    print 'val', val[0:2,0:2], 'val end', val[-3:,-3:], kv.rank




go_concat(concat_stype)
#go_sum(sum_stype)
#go_concat(concat_stype)
#go_sum(sum_stype)
if __name__ == "__main__":
    print 'haha', kv.rank

