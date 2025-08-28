import dataclasses
import datetime
import os
import timeit
from typing import Any

import numpy as np
import torch
import torch.cuda.nvtx as nvtx
import torch.distributed as dist
import torch.multiprocessing as mp

from cs336_basics import data, model as transformer, nn_utils, optimizer as opt_module

DATA_DIR = "/home/liyang2029/cs336_2025/assignment2-systems/data"


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    # dist.init_process_group("gloo", rank=rank, world_size=world_size)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def distributed_demo(rank, world_size):
    setup(rank, world_size)
    data = torch.randint(0, 10, (3,))
    print(f"rank {rank} data (before all-reduce): {data}")
    dist.all_reduce(data, async_op=False)
    print(f"rank {rank} data (after all-reduce): {data}")


def naive_ddp(rank, world_size):
    setup(rank, world_size)
    torch.manual_seed(111)
    data = torch.arange(16 * 6).view(16, 6).to(torch.float32)
    batch_size = data.shape[0] // world_size
    inputs = data[rank * batch_size : (rank + 1) * batch_size]
    model = torch.nn.Sequential(
        torch.nn.ReLU(),
        torch.nn.Linear(6, 1),
    )
    optimizer = torch.optim.Adam(model.parameters())
    for i in range(5):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = outputs.mean()
        loss.backward()
        for p in model.parameters():
            dist.all_reduce(p.grad.data, op=dist.ReduceOp.SUM)
            p.grad.data /= world_size
        optimizer.step()

    if rank == 0:
        print(f"rank {rank}")
        for name, p in model.named_parameters():
            print(name)
            print(p.data)
            print(p.grad.data)


def minimal_ddp_flat_benchmarking(
    rank: int,
    world_size: int,
    vocab_size: int = 10000,
    d_model: int = 1600,
    d_ff: int = 6400,
    num_heads: int = 25,
    num_layers: int = 48,
    context_length: int = 256,
    rope_theta: float = 10000,
    num_iters: int = 100,
    batch_size: int = 8,
    max_learning_rate: float = 1e-3,
    weight_decay: float = 0.01,
    beta1: float = 0.9,
    beta2: float = 0.95,
    eps: float = 1e-8,
    grad_l2_max: float = 7.0,
):
    setup(rank, world_size)
    device = "cuda"
    model = (
        transformer.BasicsTransformerLM(
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            rope_theta=rope_theta,
            vocab_size=vocab_size,
            context_length=context_length,
            num_layers=num_layers,
        )
        .to(device)
        .to(torch.bfloat16)
    )
    # model.compile()
    optimizer = opt_module.AdamW(
        model.parameters(),
        lr=max_learning_rate,
        weight_decay=weight_decay,
        betas=(beta1, beta2),
        eps=eps,
    )
    dataset = np.random.randint(0, vocab_size, size=32768, dtype=np.int64)
    model.train()
    for iter in range(1, num_iters + 1):
        tic1 = timeit.default_timer()
        inputs, targets = data.get_batch(dataset, batch_size, context_length, device)
        outputs = model(inputs)
        loss = nn_utils.cross_entropy(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        tic2 = timeit.default_timer()
        grads = [p.grad for p in model.parameters()]
        grads_flatten = torch._utils._flatten_dense_tensors(grads)
        dist.all_reduce(grads_flatten, op=dist.ReduceOp.SUM)
        for grad, grad_reduce in zip(
            grads, torch._utils._unflatten_dense_tensors(grads_flatten, grads)
        ):
            grad.data = grad_reduce / world_size
        torch.cuda.synchronize()
        if rank == 0:
            print(
                f"{iter} total time for communicating grads {timeit.default_timer() - tic2}"
            )
        nn_utils.clip_gradient(model.parameters(), max_norm=grad_l2_max)
        optimizer.step()
        if rank == 0:
            print(f"{iter} total time per step {timeit.default_timer() - tic1}")


class DDPIndividualParameters(torch.nn.Module):
    def __init__(self, module: torch.nn.Module):
        super().__init__()
        self.module = module
        self.handles = []

        def hook(p):
            self.handles.append(dist.all_reduce(p.grad, async_op=True))

        for p in module.parameters():
            dist.broadcast(p.data, src=0)
            if p.requires_grad:
                p.register_post_accumulate_grad_hook(hook)

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    def finish_gradient_synchronization(self):
        for h in self.handles:
            h.wait()
        self.handles.clear()
        for p in self.module.parameters():
            if p.requires_grad:
                p.grad.data /= dist.get_world_size()


def ddp_overlap_individual_parameters_benchmarking(
    rank: int,
    world_size: int,
    vocab_size: int = 10000,
    d_model: int = 1600,
    d_ff: int = 6400,
    num_heads: int = 25,
    num_layers: int = 48,
    context_length: int = 256,
    rope_theta: float = 10000,
    num_iters: int = 10,
    batch_size: int = 8,
    max_learning_rate: float = 1e-3,
    weight_decay: float = 0.01,
    beta1: float = 0.9,
    beta2: float = 0.95,
    eps: float = 1e-8,
    grad_l2_max: float = 7.0,
):
    setup(rank, world_size)
    device = "cuda"
    model = (
        transformer.BasicsTransformerLM(
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            rope_theta=rope_theta,
            vocab_size=vocab_size,
            context_length=context_length,
            num_layers=num_layers,
        )
        .to(device)
        .to(torch.bfloat16)
    )
    ddp_model = DDPIndividualParameters(model)
    # model.compile()
    optimizer = opt_module.AdamW(
        ddp_model.parameters(),
        lr=max_learning_rate,
        weight_decay=weight_decay,
        betas=(beta1, beta2),
        eps=eps,
    )
    dataset = np.random.randint(0, vocab_size, size=32768, dtype=np.int64)
    ddp_model.train()
    for iter in range(1, num_iters + 1):
        tic1 = timeit.default_timer()
        inputs, targets = data.get_batch(dataset, batch_size, context_length, device)
        with nvtx.range(f"iter {iter}"):
            outputs = ddp_model(inputs)
            loss = nn_utils.cross_entropy(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            tic2 = timeit.default_timer()
            ddp_model.finish_gradient_synchronization()
        torch.cuda.synchronize()
        if rank == 0:
            print(
                f"{iter} total time for communicating grads {timeit.default_timer() - tic2}"
            )
        nn_utils.clip_gradient(ddp_model.parameters(), max_norm=grad_l2_max)
        optimizer.step()
        if rank == 0:
            print(f"{iter} total time per step {timeit.default_timer() - tic1}")


@dataclasses.dataclass
class Bucket:
    grads: list[torch.Tensor]
    size: float = 0
    grads_flatten: torch.Tensor = None
    handle: Any = None


class DDPBucketed(torch.nn.Module):
    def __init__(self, module: torch.nn.Module, bucket_size_mb: float):
        super().__init__()
        self.module = module
        self.bucket_size = bucket_size_mb * 2**20
        self.buckets = []

        def hook(p):
            size = p.grad.data.element_size() * p.grad.data.nelement()
            if not self.buckets:
                self.buckets.append(Bucket([p.grad], size=size))
                return
            bucket = self.buckets[-1]
            if bucket.size + size <= self.bucket_size:
                bucket.grads.append(p.grad)
                bucket.size += size
                return
            bucket.grads_flatten = torch._utils._flatten_dense_tensors(bucket.grads)
            bucket.handle = dist.all_reduce(bucket.grads_flatten, async_op=True)
            self.buckets.append(Bucket([p.grad], size=size))

        for p in reversed(list(module.parameters())):
            dist.broadcast(p.data, src=0)
            if p.requires_grad:
                p.register_post_accumulate_grad_hook(hook)

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    def finish_gradient_synchronization(self):
        bucket = self.buckets[-1]
        bucket.grads_flatten = torch._utils._flatten_dense_tensors(bucket.grads)
        bucket.handle = dist.all_reduce(bucket.grads_flatten, async_op=True)
        for b in self.buckets:
            b.handle.wait()
            for grad, grad_reduce in zip(
                b.grads, torch._utils._unflatten_dense_tensors(b.grads_flatten, b.grads)
            ):
                grad.data = grad_reduce / dist.get_world_size()
        self.buckets.clear()


def ddp_bucketed_benchmarking(
    rank: int,
    world_size: int,
    vocab_size: int = 10000,
    d_model: int = 1600,
    d_ff: int = 6400,
    num_heads: int = 25,
    num_layers: int = 48,
    context_length: int = 256,
    rope_theta: float = 10000,
    num_iters: int = 10,
    batch_size: int = 8,
    max_learning_rate: float = 1e-3,
    weight_decay: float = 0.01,
    beta1: float = 0.9,
    beta2: float = 0.95,
    eps: float = 1e-8,
    grad_l2_max: float = 7.0,
):
    setup(rank, world_size)
    device = "cuda"
    model = (
        transformer.BasicsTransformerLM(
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            rope_theta=rope_theta,
            vocab_size=vocab_size,
            context_length=context_length,
            num_layers=num_layers,
        )
        .to(device)
        .to(torch.bfloat16)
    )
    ddp_model = DDPBucketed(model, bucket_size_mb=1000)
    # model.compile()
    optimizer = opt_module.AdamW(
        ddp_model.parameters(),
        lr=max_learning_rate,
        weight_decay=weight_decay,
        betas=(beta1, beta2),
        eps=eps,
    )
    dataset = np.random.randint(0, vocab_size, size=32768, dtype=np.int64)
    ddp_model.train()
    for iter in range(1, num_iters + 1):
        tic1 = timeit.default_timer()
        inputs, targets = data.get_batch(dataset, batch_size, context_length, device)
        with nvtx.range(f"iter {iter}"):
            outputs = ddp_model(inputs)
            loss = nn_utils.cross_entropy(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            tic2 = timeit.default_timer()
            ddp_model.finish_gradient_synchronization()
        torch.cuda.synchronize()
        if rank == 0:
            print(
                f"{iter} total time for communicating grads {timeit.default_timer() - tic2}"
            )
        nn_utils.clip_gradient(ddp_model.parameters(), max_norm=grad_l2_max)
        optimizer.step()
        if rank == 0:
            print(f"{iter} total time per step {timeit.default_timer() - tic1}")


if __name__ == "__main__":
    # world_size = 4
    # mp.spawn(fn=distributed_demo, args=(world_size,), nprocs=world_size, join=True)

    # world_size = 4
    # mp.spawn(fn=naive_ddp, args=(world_size,), nprocs=world_size, join=True)

    # naive_ddp(0, 1)

    world_size = 4
    mp.spawn(
        fn=ddp_bucketed_benchmarking,
        args=(world_size,),
        nprocs=world_size,
        join=True,
    )
