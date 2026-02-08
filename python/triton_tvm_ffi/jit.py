from __future__ import annotations

from functools import cached_property
import inspect
from typing import (
    Any,
    Callable,
    Dict,
    Final,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import torch
from triton.compiler import CompiledKernel
from triton.runtime import Autotuner, JITFunction
import tvm_ffi

from .utils import type_canonicalize


class TVMFFIJITFunction(object):
    def __init__(self, fn: Union[Autotuner, JITFunction], *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.fn: Final[Union[Autotuner, JITFunction]] = fn
        self.signature: List[str] = [*inspect.signature(self.basefn).parameters.keys()]
        self.best_config: Optional[Dict[str, Any]] = None
        self.ctypes: Optional[List[Optional[str]]] = None
        self.kernel: Optional[bytes] = None
        self.num_warps: Optional[int] = None
        self.shmem: int = 0

        @tvm_ffi.register_global_func(self.fullname)
        def _(
            grid: Union[
                Callable[[Dict[str, Any]], Tuple[int, int, int]], Tuple[int, int, int]
            ],
            _device: int,
            _stream: int,
            args: Sequence[Any],
            kwargs: Mapping[str, Any],
        ):
            args: Iterator[Any] = map(self.canonicalize, args)
            kwargs: Dict[str, Any] = {
                k: v for k, v in zip(self.signature, args) if v is not None
            } | {k: self.canonicalize(v) for k, v in kwargs.items()}
            kernel: CompiledKernel = self.fn[grid](*args, **kwargs)
            self.num_warps, _, self.shmem = kernel.packed_metadata
            self.ctypes = [type_canonicalize(v) for v in kernel.src.signature.values()]
            self.kernel = kernel.kernel
            if isinstance(self.fn, Autotuner):
                self.best_config = self.fn.best_config.all_kwargs()
            return kernel

    def __getitem__(
        self,
        grid: Union[
            Callable[[Dict[str, Any]], Tuple[int, int, int]], Tuple[int, int, int]
        ],
    ):
        return self.fn[grid]

    @cached_property
    def basefn(self) -> Callable:
        return self.jitfn.fn

    @property
    def cache_hash(self) -> int:
        return self.ctypes_hash ^ self.kernel_hash

    @property
    def ctypes_hash(self) -> int:
        return hash(tuple(self.ctypes) if self.ctypes is not None else None)

    @cached_property
    def fnname(self) -> str:
        return self.basefn.__name__

    @cached_property
    def fullname(self) -> str:
        return f"triton.{self.name}"

    @cached_property
    def jitfn(self) -> JITFunction:
        fn: Union[Autotuner, JITFunction] = self.fn
        while not isinstance(fn, JITFunction):
            fn = fn.fn
        return fn

    @property
    def kernel_hash(self) -> int:
        return hash(self.kernel)

    @property
    def kernel_cstr(self) -> Optional[str]:
        if self.kernel is not None:
            return "".join(f"\\x{byte:02x}" for byte in self.kernel)
        else:
            return None

    @cached_property
    def name(self) -> str:
        return f"{self.fnname}_{hash(self.basefn)}"

    @staticmethod
    def canonicalize(val: Any) -> Any:
        if hasattr(val, "__dlpack__"):
            return torch.from_dlpack(val)
        else:
            return val


def jit(fn: JITFunction) -> TVMFFIJITFunction:
    return TVMFFIJITFunction(fn)
