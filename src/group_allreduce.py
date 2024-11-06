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

def split_all_reduce(tensor, groups):
    # For now, assert 1-D tensor. We can flatten in future for multi-dimensional support.
    assert(len(tensor.shape) == 1)
    reqs = []

    ngroups = len(groups)

    split_size = (tensor.shape[0] + ngroups - 1) // ngroups

    for i,group in enumerate(groups):
        subtensor = tensor[i*split_size:(i+1)*split_size]
        req = dist.all_reduce(subtensor, group=group, async_op=True)
        reqs.append(req)

    for req in reqs:
        req.wait()

    dist.barrier()

os.environ["MASTER_ADDR"] = "127.0.0.1"
os.environ["MASTER_PORT"] = "29500"
os.environ["RANK"] = get_rank_from_env()
os.environ["WORLD_SIZE"] = get_nprocs_from_env()
dist.init_process_group(backend="ccl", init_method="env://")

nbytes = 1000*1000*1000

n = nbytes // 4
nbytes = n * 4
gbytes = nbytes / 1000 / 1000 / 1000

tensor = torch.ones(n, dtype=torch.float32, device=get_device())

dummy_tensor = torch.zeros(n, dtype=torch.float32, device=get_device())
dist.all_reduce(dummy_tensor)

if dist.get_rank() == 0:
    print('Execute with a single all_reduce kernel:')

n_iterations = 20

for i in range(n_iterations):
    dist.barrier()
    begin = time.time()

    dist.all_reduce(tensor)

    end = time.time()

    duration = end - begin

    if dist.get_rank() == 0:
        print('Took %s ms, achieved %s GB/s' % (duration*1000, gbytes / duration))

devices = list(range(dist.get_world_size()))
devices_backwards = list(devices)
devices_backwards.reverse()

split = tensor.shape[0] // 2


forward_group = dist.new_group(ranks=devices)
backward_group = dist.new_group(ranks=devices_backwards)

if dist.get_rank() == 0:
    print('Creating two process groups:')
    print('devices: %s' % (devices,))
    print('devices_backwards: %s' % (devices_backwards,))

tensor = torch.ones(n, dtype=torch.float32, device=get_device())

if dist.get_rank() == 0:
    print('Double AllReduce:')

for i in range(n_iterations):
    dist.barrier()
    begin = time.time()

    req1 = dist.all_reduce(tensor[:split], group=forward_group, async_op=True)
    req2 = dist.all_reduce(tensor[split:], group=backward_group, async_op=True)

    req1.wait()
    req2.wait()

    dist.barrier()
    end = time.time()

    duration = end - begin

    if dist.get_rank() == 0:
        print('Took %s ms, achieved %s GB/s' % (duration*1000, gbytes / duration))

if dist.get_rank() == 0:
    print('6-Ary AllReduce:')

device_lists = [
  [0, 1, 2, 3, 4, 5, 6, 7],
  [0, 2, 1, 3, 5, 4, 7, 6],
  [0, 4, 1, 5, 7, 2, 6, 3],
  [0, 5, 1, 7, 3, 6, 4, 2],
  [0, 6, 5, 2, 4, 3, 7, 1]
]

groups = []

for device_list in device_lists:
    group = dist.new_group(ranks=device_list)
    groups.append(group)

for i in range(n_iterations):
    tensor = torch.ones(n, dtype=torch.float32, device=get_device())
    dist.barrier()
    begin = time.time()

    split_all_reduce(tensor, groups)

    end = time.time()
    duration = end - begin

    assert(torch.sum(tensor - dist.get_world_size()*torch.ones(n, dtype=torch.float32, device=get_device())) == 0)

    if dist.get_rank() == 0:
        print('Took %s ms, achieved %s GB/s' % (duration*1000, gbytes / duration))
