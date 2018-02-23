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
#sys.path.insert(0, "../../python/")
import mxnet as mx
import numpy as np
import numpy.random as rnd


keys = ['3','5','9']
sum_keys = ['sum1','sum2']
concat_keys = ['concat1','concat2']
shape = (2,3)
big_shape = (400, 400)
rate = 1
stype = 'sum_single'

kv = mx.kv.create('dist_sync')

def init_kv():
    # init kv dns keys
    kv.init(keys, [mx.nd.ones(shape)] * len(keys))
    kv.init('99', mx.nd.ones(big_shape))
    # worker info
    my_rank = kv.rank
    nworker = kv.num_workers
    # init updater on servers
    kv.set_optimizer(mx.optimizer.create('test', rescale_grad=rate))
    return kv, my_rank, nworker

def init_kv_special_single():
    # init kv dns keys
    kv.init_kvspecial('sum1', mx.nd.ones(shape), stype)
    kv.init_kvspecial('sum2', mx.nd.ones(big_shape), stype)
    # worker info
    my_rank = kv.rank
    nworker = kv.num_workers
    return kv, my_rank, nworker

def init_kv_special_multi():
    # init kv dns keys
    kv.init_kvspecial('sum1', mx.nd.ones(shape), 'sum_alone')
    kv.init_kvspecial('sum2', mx.nd.ones(big_shape), 'sum')
    kv.init_kvspecial('concat1', mx.nd.ones(shape), 'concat_alone')
    # worker info
    my_rank = kv.rank
    nworker = kv.num_workers
    return kv, my_rank, nworker



def test_sync_push_pull_special():
    kv, my_rank, nworker = init_kv_special_single(sum_keys)
    def check_default_keys(kv, my_rank, nworker):
        nrepeat = 3
        # checks pull after push in loop, because behavior during
        # consecutive pushes doesn't offer any guarantees
        for i in range(nrepeat):
            kv.push_kvspecial('sum1', mx.nd.ones(shape)*(my_rank+1), stype)
            kv.push_kvspecial('sum2', mx.nd.ones(big_shape)*(my_rank+1), stype)
            num = (nworker + 1) * nworker * rate / 2 * (i + 1) + 1
            val = mx.nd.zeros(shape)
            kv.pull_kvspecial('sum1', val, stype)
            print 'val', val
            val2 = mx.nd.zeros(big_shape)
            kv.pull_kvspecial('sum2', val2, stype)
            print 'val2', val2

if __name__ == "__main__":
    test_sync_push_pull()
