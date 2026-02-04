from __future__ import annotations

from functools import cached_property
from typing import Any, Dict, Final, List, Optional, Tuple

import torch
from triton.compiler import CompiledKernel
from triton.runtime import JITFunction
import tvm_ffi

from .utils import type_canonicalize


class TVMFFIJITFunction(object):
    def __init__(self, fn: JITFunction, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.fn: Final[JITFunction] = fn
        self.ctypes: Optional[List[Optional[str]]] = None
        self.kernel: Optional[bytes] = None
        self.num_warps: Optional[int] = None

        @tvm_ffi.register_global_func(self.fullname)
        def _(
            grid: Tuple[int, int, int],
            _: int,
            num_warps: Optional[int],
            num_stages: Optional[int],
            *args,
            **kwargs,
        ):
            args: List[Any] = map(self.canonicalize, args)
            kwargs: Dict[str, Any] = {
                k: self.canonicalize(v) for k, v in kwargs.items()
            }
            if num_warps is not None:
                kwargs["num_warps"] = num_warps
            if num_stages is not None:
                kwargs["num_stages"] = num_stages
            kernel: CompiledKernel = self.fn[grid](*args, **kwargs)
            self.num_warps, _, _ = kernel.packed_metadata
            self.ctypes = [type_canonicalize(v) for v in kernel.src.signature.values()]
            self.kernel = kernel.kernel
            return kernel

    def __getitem__(self, grid: Tuple[int, int, int]):
        return self.fn[grid]

    @property
    def cache_hash(self) -> int:
        return self.ctypes_hash ^ self.kernel_hash

    @property
    def ctypes_hash(self) -> int:
        return hash(tuple(self.ctypes) if self.ctypes is not None else None)

    @property
    def kernel_hash(self) -> int:
        return hash(self.kernel)

    @cached_property
    def fnname(self) -> str:
        return self.fn.fn.__name__

    @cached_property
    def fullname(self) -> str:
        return f"triton.{self.name}"

    @cached_property
    def name(self) -> str:
        return f"{self.fnname}_{hash(self.fn.fn)}"

    @staticmethod
    def canonicalize(val: Any) -> Any:
        if hasattr(val, "__dlpack__"):
            return torch.from_dlpack(val)
        else:
            return val


def jit(fn: JITFunction) -> TVMFFIJITFunction:
    return TVMFFIJITFunction(fn)
