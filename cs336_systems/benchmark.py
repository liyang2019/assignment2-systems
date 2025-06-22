import torch
import timeit
import numpy as np
import torch.cuda.nvtx as nvtx
from torch import Tensor
from einops import einsum
from jaxtyping import Float, Bool, Int
import math
from torch import nn
import pandas as pd
import os
import time

import cs336_basics.model
import cs336_basics.nn_utils
import cs336_basics.optimizer


@nvtx.range("scaled dot product attention")
def annotated_scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys    d_k"],
    V: Float[Tensor, " ... keys    d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    """Scaled dot-product attention.

    This function implements Eq. 1 of the Transformer paper.

    Args:
        Q: Tensor of queries, may have any number of leading dimensions.
        K: Tensor of keys, sharing leading dimensions with Q.
        V: Tensor of values, sharding leading dimensions with Q and K.
        mask: An (optional) mask of shape (..., seq_len, seq_len).
            Attention scores for positions with a mask value of `False` should
            be masked out, i.e., not affect the softmaxed attention probabilities.

    Returns:
        torch.FloatTensor of shape (..., seq_len, value_dimension)
        with the output of running your scaled dot product attention
        implementation with the provided key, query, and value tensors.
    """

    d_k = K.shape[-1]
    with nvtx.range("computing attention scores"):
        # FLOPs: 2BHLDL
        attention_scores = einsum(
            Q, K, "... query d_k, ... key d_k -> ... query key"
        ) / math.sqrt(d_k)

    with nvtx.range("computing softmax"):
        if mask is not None:
            attention_scores = torch.where(mask, attention_scores, float("-inf"))

        # FLOPs: 2BHLL
        attention_weights = cs336_basics.model.softmax(
            attention_scores, dim=-1
        )  # Softmax over the key dimension

    with nvtx.range("final matmul"):
        # FLOPs: 2BHLLD
        outputs = einsum(
            attention_weights, V, "... query key, ... key d_v ->  ... query d_v"
        )
    return outputs


cs336_basics.model.scaled_dot_product_attention = annotated_scaled_dot_product_attention


def basic_benchmark(
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    context_length: int,
    rope_theta: float = 10000,
    vocab_size: int = 10000,
    batch_size: int = 4,
    warmup_steps: int = 5,
    benchmark_steps: int = 10,
    device: str = "cuda:0",
    use_bf16: bool = False,
    run_backward: bool = True,
    memory_snapshot_file: str = "memory_snapshot.pickle",
):
    m = cs336_basics.model.BasicsTransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        rope_theta=rope_theta,
    ).to(device)
    num_parameters = sum(np.prod(p.shape) for p in m.parameters())
    print(f"parameters count {num_parameters}")
    print(f"parameters memory {num_parameters * 4 / 2**30} GiB")
    optimizer = cs336_basics.optimizer.AdamW(m.parameters())
    x = torch.randint(0, vocab_size, (batch_size, context_length)).to(device)
    for _ in range(warmup_steps):
        optimizer.zero_grad()
        with torch.autocast("cuda", dtype=torch.float16, enabled=use_bf16):
            y = m(x)
            if run_backward:
                loss = cs336_basics.nn_utils.cross_entropy(y, x)
        if run_backward:
            loss.backward()
            optimizer.step()
    torch.cuda.synchronize()

    torch.cuda.memory._record_memory_history(max_entries=1000000)

    with nvtx.range("forward & backward"):
        forward_times = []
        backward_times = []
        for _ in range(benchmark_steps):
            optimizer.zero_grad()
            tic = timeit.default_timer()
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_bf16):
                with nvtx.range("forward"):
                    y = m(x)
            torch.cuda.synchronize()
            forward_times.append(timeit.default_timer() - tic)

            if run_backward:
                with torch.autocast("cuda", dtype=torch.float16, enabled=use_bf16):
                    with nvtx.range("loss"):
                        loss = cs336_basics.nn_utils.cross_entropy(y, x)
                    torch.cuda.synchronize()

                tic = timeit.default_timer()
                with nvtx.range("backward"):
                    loss.backward()
                    optimizer.step()
                    torch.cuda.synchronize()
                backward_times.append(timeit.default_timer() - tic)

    torch.cuda.memory._dump_snapshot(memory_snapshot_file)
    torch.cuda.memory._record_memory_history(enabled=None)

    print(
        f"forward mean {np.mean(forward_times) * 1000:>6.2f} ms "
        f"std {np.std(forward_times)  * 1000:>6.2f} ms",
        end=" ",
    )
    if run_backward:
        print(
            f"backward mean {np.mean(backward_times) * 1000:>6.2f} ms "
            f"std {np.std(backward_times)  * 1000:>6.2f} ms",
            end=" ",
        )
    print()


def run_basic_benchmark():
    model_sizes = {
        # "small": {
        #     "d_model": 768,
        #     "d_ff": 3072,
        #     "num_layers": 12,
        #     "num_heads": 12,
        # },
        # "medium": {
        #     "d_model": 1024,
        #     "d_ff": 4096,
        #     "num_layers": 24,
        #     "num_heads": 16,
        # },
        "large": {
            "d_model": 1280,
            "d_ff": 5120,
            "num_layers": 36,
            "num_heads": 20,
        },
        # "xl": {
        #     "d_model": 1600,
        #     "d_ff": 6400,
        #     "num_layers": 48,
        #     "num_heads": 25,
        # },
        # "2.7B": {
        #     "d_model": 2560,
        #     "d_ff": 10240,
        #     "num_layers": 32,
        #     "num_heads": 32,
        # },
    }
    for name, sizes in model_sizes.items():
        for run_backward in [
            True,
            False,
        ]:
            for context_length in [
                # 32,
                64,
                128,
                256,
                # 512,
            ]:
                for use_bf16 in [True, False]:
                    print(
                        f"{name} "
                        f"bf16 {use_bf16} "
                        f"context_len {context_length} "
                        f"backward {run_backward}"
                    )
                    basic_benchmark(
                        **sizes,
                        context_length=context_length,
                        warmup_steps=5,
                        benchmark_steps=1,
                        use_bf16=use_bf16,
                        run_backward=run_backward,
                        memory_snapshot_file=(
                            f"memory_snapshot-{name}"
                            f"-{context_length}"
                            f"-backward{int(run_backward)}"
                            f"-bf16{int(use_bf16)}"
                            ".pickle"
                        ),
                    )
                    torch.cuda.empty_cache()


def memory_results():
    records = [
        {"context_length": 64, "bf16": False, "backward": False, "max_memory": 6.5},
        {"context_length": 64, "bf16": True, "backward": False, "max_memory": 9.0},
        {"context_length": 64, "bf16": False, "backward": True, "max_memory": 14.5},
        {"context_length": 64, "bf16": True, "backward": True, "max_memory": 14.5},
        {"context_length": 128, "bf16": False, "backward": False, "max_memory": 9.7},
        {"context_length": 128, "bf16": True, "backward": False, "max_memory": 11.0},
        {"context_length": 128, "bf16": False, "backward": True, "max_memory": 14.5},
        {"context_length": 128, "bf16": True, "backward": True, "max_memory": 14.5},
        {"context_length": 256, "bf16": False, "backward": False, "max_memory": 17.1},
        {"context_length": 256, "bf16": True, "backward": False, "max_memory": 15.9},
        {"context_length": 256, "bf16": False, "backward": True, "max_memory": 17.7},
        {"context_length": 256, "bf16": True, "backward": True, "max_memory": 17.1},
    ]
    return pd.DataFrame.from_records(records)


def attention_benchmark():
    device = "cuda"
    torch.manual_seed(123)
    os.makedirs("memory_snapshot_attn", exist_ok=True)
    stats = []
    torch.cuda.memory._record_memory_history(max_entries=1000000)
    fun = torch.compile(annotated_scaled_dot_product_attention)
    torch._functorch.config.donated_buffer=False
    for d_head in [16, 32, 64, 128]:
        for context_length in [
            256,
            1024,
            4096,
            8192,
            # 16384,
        ]:
            q = torch.rand(8, context_length, d_head, device=device, requires_grad=True)
            k = torch.rand(8, context_length, d_head, device=device, requires_grad=True)
            v = torch.rand(8, context_length, d_head, device=device, requires_grad=True)

            record = {
                "L": context_length,
                "D": d_head,
            }

            # test foward
            # warmup
            for i in range(5):
                loss = fun(q, k, v).sum()
            torch.cuda.synchronize()

            forward_times = []
            for i in range(100):
                tic = timeit.default_timer()
                loss = fun(q, k, v).sum()
                torch.cuda.synchronize()
                forward_times.append(timeit.default_timer() - tic)
            record["forward_time"] = np.mean(forward_times) * 1000
            forward_mem = torch.cuda.memory.memory_allocated(device)
            record["forward_mem"] = forward_mem

            # test backward
            # warmup
            for i in range(5):
                loss.backward(retain_graph=True)
            torch.cuda.synchronize()

            backward_times = []
            for i in range(100):
                tic = timeit.default_timer()
                loss.backward(retain_graph=True)
                torch.cuda.synchronize()
                backward_times.append(timeit.default_timer() - tic)
            record["backward_time"] = np.mean(backward_times) * 1000
            backward_mem = torch.cuda.memory.memory_allocated(device)
            record["backward_mem"] = backward_mem - forward_mem

            print(
                " ".join(
                    (
                        f"{k:<10} {v:<10.4f}"
                        if isinstance(v, float)
                        else f"{k:<10} {v:<10}"
                    )
                    for k, v in record.items()
                )
            )
            stats.append(record)

            # torch.cuda.empty_cache()

    torch.cuda.memory._record_memory_history(enabled=None)

    df = pd.DataFrame.from_records(stats)
    df.to_csv("attention_stats_compile.csv")


def mixed_precision_accumulation():
    s = torch.tensor(0, dtype=torch.float32)
    for i in range(1000):
        s += torch.tensor(0.01, dtype=torch.float32)
    print(s)
    s = torch.tensor(0, dtype=torch.float16)
    for i in range(1000):
        s += torch.tensor(0.01, dtype=torch.float16)
    print(s)
    s = torch.tensor(0, dtype=torch.float32)
    for i in range(1000):
        s += torch.tensor(0.01, dtype=torch.float16)
    print(s)
    s = torch.tensor(0, dtype=torch.float32)
    for i in range(1000):
        x = torch.tensor(0.01, dtype=torch.float16)
        s += x.type(torch.float32)
    print(s)


class ToyModel(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 10, bias=False)
        self.ln = nn.LayerNorm(10)
        self.fc2 = nn.Linear(10, out_features, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        print("parameter dtypes")
        for p in self.parameters():
            print(p.dtype)
        print(f"input dtype {x.dtype}")
        x = self.fc1(x)
        print(f"fc1(x) dtype {x.dtype}")
        x = self.relu(x)
        print(f"relu(x) dtype {x.dtype}")
        x = self.ln(x)
        print(f"ln(x) dtype {x.dtype}")
        x = self.fc2(x)
        print(f"fc2(x) dtype {x.dtype}")
        return x


def run_autocast_toy_model():
    m = ToyModel(2, 3).cuda()
    inputs = torch.rand((3, 2)).cuda()
    targets = torch.rand((3, 3)).cuda()

    with torch.autocast("cuda", dtype=torch.float16):
        outputs = m(inputs)
        print(f"output dtype {outputs.dtype}")
        loss = ((outputs - targets) ** 2.0).mean()
        print(f"loss dtype {loss.dtype}")
    loss.backward()
    print("grad dtypes")
    for p in m.parameters():
        print(p.grad.dtype)


if __name__ == "__main__":
    # uv run nsys profile -o result --python-backtrace=cuda --force-overwrite true python cs336_systems/benchmark.py
    # run_basic_benchmark()
    attention_benchmark()
    # mixed_precision_accumulation()
    # run_autocast_toy_model()
