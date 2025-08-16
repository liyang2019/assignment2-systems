import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


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
    inputs = data[rank * batch_size: (rank + 1) * batch_size]
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


def naive_ddp_benchmarking(rank, world_size):
    


if __name__ == "__main__":
    world_size = 4
    mp.spawn(fn=distributed_demo, args=(world_size,), nprocs=world_size, join=True)

    world_size = 4
    mp.spawn(fn=naive_ddp, args=(world_size,), nprocs=world_size, join=True)

    naive_ddp(0, 1)
