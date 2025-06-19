import torch
import timeit
import numpy as np
import torch.cuda.nvtx as nvtx

from cs336_basics import model


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
    test_backward: bool = True,
    device: str = "cuda:0",
):
    m = model.BasicsTransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        rope_theta=rope_theta,
    ).to(device)
    x = torch.randint(0, vocab_size, (batch_size, context_length)).to(device)
    for _ in range(warmup_steps):
        y = m(x)
        if test_backward:
            y.mean().backward()

    with nvtx.range("forward & backward"):
        forward_times = []
        backward_times = []
        for _ in range(benchmark_steps):
            tic = timeit.default_timer()
            with nvtx.range("forward"):
                y = m(x)
                torch.cuda.synchronize()
            forward_times.append(timeit.default_timer() - tic)
            if test_backward:
                tic = timeit.default_timer()
                with nvtx.range("backward"):
                    y.mean().backward()
                    torch.cuda.synchronize()
                backward_times.append(timeit.default_timer() - tic)
    print(
        f"forward mean {np.mean(forward_times):.6f} std {np.std(forward_times):.6f}",
        end=" ",
    )
    print(
        f"backward mean {np.mean(backward_times):.6f} std {np.std(backward_times):.6f}"
    )


if __name__ == "__main__":
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
        print(name)
        basic_benchmark(
            **sizes,
            context_length=32,
            warmup_steps=0,
        )
