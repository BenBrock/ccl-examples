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
nbytes = n * 4
gbytes = nbytes / 1000 / 1000 / 1000

send_tensor = torch.zeros(n, dtype=torch.float32, device=get_device())
recv_tensor = torch.zeros(n, dtype=torch.float32, device=get_device())

# Perform an all_reduce to initialize communicators and such.
dist.all_reduce(send_tensor)

durations = defaultdict(list)

for send_rank in range(dist.get_world_size()):
    for recv_rank in range(dist.get_world_size()):
        if send_rank != recv_rank:
            dist.barrier()
            begin = time.time()

            reqs = []

            if dist.get_rank() == send_rank:
                req = dist.isend(send_tensor, recv_rank)
                reqs.append(req)

            if dist.get_rank() == recv_rank:
                req = dist.irecv(recv_tensor, send_rank)
                reqs.append(req)

            for req in reqs:
                req.wait()

            dist.barrier()
            end = time.time()
            duration = end - begin

            if send_rank == recv_rank:
                durations['same_tile'].append(duration)
            elif send_rank // 2 == recv_rank // 2:
                durations['same_device'].append(duration)
            else:
                durations['different_device'].append(duration)

            if dist.get_rank() == recv_rank:
                print('%s -> %s took %s ms, achieved %s GB/s' % (send_rank, recv_rank, 1000*duration, gbytes / duration))

dist.barrier()

for transfer_type,durs in durations.items():
    if dist.get_rank() == 0:
        print('On average, %s took %s ms, achieved %s GB/s'% (transfer_type, 1000*np.mean(durs), gbytes / np.mean(durs)))
