from pathlib import Path
import time

import torch
import triton
import triton.language as tl
import triton_tvm_ffi

DEVICE = triton.runtime.driver.active.get_active_torch_device()


@triton_tvm_ffi.jit
@triton.jit
def add_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)


def add_triton(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    output: torch.Tensor = torch.empty_like(x)
    assert x.device == DEVICE and y.device == DEVICE and output.device == DEVICE
    n_elements: int = output.numel()
    BLOCK_SIZE: int = 1024
    add_kernel[lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), 1, 1)](
        x, y, output, n_elements, BLOCK_SIZE
    )
    return output


@triton_tvm_ffi.torch_wrap(
    [add_kernel],
    Path(__file__).parent / "add.cc",
)
def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor: ...


if __name__ == "__main__":
    torch.manual_seed(0)
    size = 98432
    x = torch.rand(size, device=DEVICE)
    y = torch.rand(size, device=DEVICE)
    output_torch = x + y
    output_triton = add_triton(x, y)
    output_tvm_ffi = add(x, y)
    assert torch.allclose(output_torch, output_triton)
    assert torch.allclose(output_torch, output_tvm_ffi)
    output_tvm_ffi = add(x, y)
    assert torch.allclose(output_torch, output_tvm_ffi)

    round = 1000
    cp0 = time.perf_counter_ns()
    for _ in range(round):
        x + y
    cp1 = time.perf_counter_ns()
    for _ in range(round):
        add_triton(x, y)
    cp2 = time.perf_counter_ns()
    for _ in range(round):
        add(x, y)
    cp3 = time.perf_counter_ns()
    print(
        f"PyTorch: {(cp1 - cp0) / round * 1e-6:.3f} ms\nTriton: {(cp2 - cp1) / round * 1e-6:.3f} ms\nTVM FFI: {(cp3 - cp2) / round * 1e-6:.3f} ms"
    )
