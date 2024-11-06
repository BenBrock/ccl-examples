#!/usr/bin/env python

import os
import sys
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

n = 4*1024*1024 // 4
nbytes = n * 4
gibytes = nbytes / 1024 / 1024 / 1024
tensor = torch.zeros(n, dtype=torch.float32, device=get_device())

# Perform an all_reduce to initialize communicators and such.
dist.all_reduce(tensor)

for rank in range(dist.get_world_size()):
    if rank == dist.get_rank():
        print('Process %s / %s is using device %s' % (dist.get_rank(), dist.get_world_size(), get_device()))
        print('  Process %s sees the following devices:', (dist.get_rank(),))

        for i in range(torch.xpu.device_count()):
            print('    - %s', (torch.xpu.get_device_properties(i),))
    dist.barrier()
