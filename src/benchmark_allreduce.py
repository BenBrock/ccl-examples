#!/usr/bin/env python

import os
import sys
import time
import torch
import torch.distributed as dist
import intel_extension_for_pytorch as ipex
import oneccl_bindings_for_pytorch as torch_ccl

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
nbytes = n * 4
gbytes = nbytes / 1000 / 1000 / 1000

tensor = torch.zeros(n, dtype=torch.float32, device=get_device())

# Perform an all_reduce to initialize communicators and such.
dist.all_reduce(tensor)

n_iterations = 10

for i in range(n_iterations):
  dist.barrier()
  begin = time.time()
  dist.all_reduce(tensor)
  end = time.time()

  duration = end - begin
  gbytes_s = gbytes / duration

  if dist.get_rank() == 0:
      print('AllReduce took %s s (achieved %s GB/s)' % (duration, gbytes_s))
