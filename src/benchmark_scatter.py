#!/usr/bin/env python

import os
import sys
import time
import numpy as np
import torch
import torch.distributed as dist
import intel_extension_for_pytorch as ipex
import oneccl_bindings_for_pytorch as torch_ccl

from collections import defaultdict

def get_device():
    return 'xpu:%s' % (dist.get_rank() % torch.xpu.device_count(),)

def get_rank_from_env():
    if 'PMI_RANK' in os.environ:
        return os.environ['PMI_RANK']
    elif 'PMIX_RANK' in os.environ:
        return os.environ['PMIX_RANK']
    elif 'RANK' in os.environ:
        return os.environ['RANK']
    else:
        raise Exception('Error: neither \'PMI_RANK\' nor \'RANK\' environment variable found. Are you invoking this script using mpirun or torchrun?')

def get_nprocs_from_env():
    if 'PMI_SIZE' in os.environ:
        return os.environ['PMI_SIZE']
    elif 'WORLD_SIZE' in os.environ:
        return os.environ['WORLD_SIZE']
    else:
        raise Exception('Error: neither \'PMI_SIZE\' nor \'WORLD_SIZE\' environment variable found. Are you invoking this script using mpirun or torchrun?')

os.environ["MASTER_ADDR"] = "127.0.0.1"
os.environ["MASTER_PORT"] = "29500"
os.environ["RANK"] = get_rank_from_env()
os.environ["WORLD_SIZE"] = get_nprocs_from_env()
dist.init_process_group(backend="ccl", init_method="env://")

nbytes = 1000*1000*1000

n = nbytes // 4

local_n = (n + dist.get_world_size() - 1) // dist.get_world_size()

n = local_n * dist.get_world_size()
nbytes = n * 4

gbytes = nbytes / 1000 / 1000 / 1000

dest_tensor = torch.zeros(local_n, dtype=torch.float32, device=get_device())
source_tensor = torch.zeros(n, dtype=torch.float32, device=get_device())

tensor_list = []

for i in range(dist.get_world_size()):
    tensor_list.append(source_tensor[i*local_n:(i+1)*local_n])

# Perform an all_reduce to initialize communicators and such.
dist.all_reduce(dest_tensor)

durations = []

for source_rank in range(dist.get_world_size()):
    dist.barrier()
    begin = time.time()

    if dist.get_rank() == source_rank:
        dist.scatter(dest_tensor, scatter_list=tensor_list, src=source_rank)
    else:
        dist.scatter(dest_tensor, scatter_list=None, src=source_rank)

    dist.barrier()
    end = time.time()

    duration = end - begin

    durations.append(duration)

    if dist.get_rank() == source_rank:
        print('Scatter %s -> took %s ms, achieved %s GB/s' % (source_rank, 1000*duration, gbytes / duration))

if dist.get_rank() == 0:
    print('On average, took %s ms, achieved %s GB/s'% (1000*np.mean(durations), gbytes / np.mean(durations)))
