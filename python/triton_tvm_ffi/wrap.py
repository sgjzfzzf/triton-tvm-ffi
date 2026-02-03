from functools import cached_property
from io import TextIOWrapper
from pathlib import Path
from typing import Final, List, Optional, Sequence, Tuple, Union

import torch.utils.cpp_extension
import tvm_ffi

from .jit import TVMFFIJITFunction


class TVMFFIWrapperFunction(object):
    def __init__(
        self,
        name: str,
        fns: List[TVMFFIJITFunction],
        code: Union[str, Path, TextIOWrapper],
        extra_cflags: Optional[Sequence[str]] = None,
        extra_cuda_cflags: Optional[Sequence[str]] = None,
        extra_ldflags: Optional[Sequence[str]] = None,
        extra_include_paths: Optional[Sequence[Union[str, Path]]] = None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.name: Final[str] = name
        self.fns: List[TVMFFIJITFunction] = [*fns]
        if isinstance(code, Path):
            with open(code, "r") as f:
                self.code: Final[str] = f.read()
        elif isinstance(code, TextIOWrapper):
            self.code: Final[str] = code.read()
        else:
            self.code: Final[str] = f"{code}"
        self.extra_cflags: Optional[Sequence[str]] = extra_cflags
        self.extra_cuda_cflags: Optional[Sequence[str]] = extra_cuda_cflags
        self.extra_ldflags: Optional[Sequence[str]] = extra_ldflags
        self.extra_include_paths: Optional[Sequence[Union[str, Path]]] = (
            extra_include_paths
        )

    def __call__(self, *args, **kwargs) -> None:
        func: tvm_ffi.Function = self.compile()
        return func(*args, **kwargs)

    @property
    def fns_hash(self) -> int:
        return hash(tuple(fn.cache_hash for fn in self.fns))

    @cached_property
    def fullname(self) -> str:
        return f"triton.{self.name}"

    @property
    def emit(self) -> str:
        defs: str = "\n".join(
            [
                "#include <cuda.h>",
                "#include <tvm/ffi/extra/cuda/cubin_launcher.h>",
                "#include <tvm/ffi/function.h>",
                f'#define {self.name.upper()}_NAME "{self.uniquename}"',
                *map(
                    self.gendef,
                    self.fns,
                ),
            ]
        )
        return f"{defs}\n{self.code}"

    @property
    def uniquename(self) -> str:
        return f"{self.name}_{self.fns_hash}"

    def compile(self) -> tvm_ffi.Function:
        if func := tvm_ffi.get_global_func(self.uniquename, allow_missing=True):
            return func
        else:
            tvm_ffi.cpp.load_inline(
                self.name,
                cpp_sources=[self.emit],
                extra_cflags=self.extra_cflags,
                extra_cuda_cflags=self.extra_cuda_cflags,
                extra_ldflags=self.extra_ldflags,
                extra_include_paths=self.extra_include_paths,
                embed_cubin={
                    f"triton_{fn.fnname}": fn.kernel
                    for fn in self.fns
                    if fn.kernel is not None
                },
            )
            return tvm_ffi.get_global_func(self.uniquename, allow_missing=True)

    @staticmethod
    def gendef(fn: TVMFFIJITFunction) -> str:
        if fn.ctypes is None:
            return f'#define {fn.fnname.upper()}_STUB tvm::ffi::Function::GetGlobalRequired("{fn.fullname}")'
        else:
            ctype_arg_list: List[Tuple[str, str]] = [
                (ctype, f"__arg{idx}") for idx, ctype in enumerate(fn.ctypes)
            ]

            return """
TVM_FFI_EMBED_CUBIN(triton_{fnname});
#define {}_STUB(__gtuple, __stream, __numWarps, __numStages, {}) do {{ \\
static auto __kernel = TVM_FFI_EMBED_CUBIN_GET_KERNEL(triton_{fnname}, "{fnname}"); \\
tvm::ffi::dim3 __grid(__gtuple.get<0>(), __gtuple.get<1>(), __gtuple.get<2>()); \\
tvm::ffi::dim3 __block(__numWarps * 32, 1, 1); \\
void *dummy = nullptr, {}; \\
void *__params[] = {{{}, &dummy, &dummy}}; \\
TVM_FFI_CHECK_CUBIN_LAUNCHER_CUDA_ERROR(__kernel.Launch(__params, __grid, __block, static_cast<tvm::ffi::cuda_api::StreamHandle>(__stream))); \\
}} while (false)
""".format(
                fn.fnname.upper(),
                ", ".join(arg for _, arg in ctype_arg_list),
                ", ".join(
                    f"*{arg}_ptr = {arg}.data_ptr()"
                    for ctype, arg in ctype_arg_list
                    if ctype == "CUdeviceptr"
                ),
                ", ".join(
                    f"&{arg}" if ctype != "CUdeviceptr" else f"&{arg}_ptr"
                    for ctype, arg in ctype_arg_list
                    if ctype is not None
                ),
                fnname=fn.fnname,
            ).strip()


def wrap(
    name: str,
    fns: List[TVMFFIJITFunction],
    code: Union[str, Path, TextIOWrapper],
    extra_cflags: Optional[Sequence[str]] = None,
    extra_cuda_cflags: Optional[Sequence[str]] = None,
    extra_ldflags: Optional[Sequence[str]] = None,
    extra_include_paths: Optional[Sequence[Union[str, Path]]] = None,
) -> TVMFFIWrapperFunction:
    return TVMFFIWrapperFunction(
        name,
        fns,
        code,
        extra_cflags,
        extra_cuda_cflags,
        extra_ldflags,
        extra_include_paths,
    )


def torch_wrap(
    name: str,
    fns: List[TVMFFIJITFunction],
    code: Union[str, Path, TextIOWrapper],
    extra_cflags: Optional[Sequence[str]] = None,
    extra_cuda_cflags: Optional[Sequence[str]] = None,
    extra_ldflags: Optional[Sequence[str]] = None,
    extra_include_paths: Optional[Sequence[Union[str, Path]]] = None,
) -> TVMFFIWrapperFunction:
    return wrap(
        name,
        fns,
        code,
        extra_ldflags=[
            "-Wl,--no-as-needed",
            *map(
                lambda path: f"-L{path}",
                torch.utils.cpp_extension.library_paths(),
            ),
            "-lc10",
            "-ltorch",
        ]
        + (extra_ldflags or []),
        extra_cflags=extra_cflags,
        extra_cuda_cflags=extra_cuda_cflags,
        extra_include_paths=[*torch.utils.cpp_extension.include_paths()]
        + (extra_include_paths or []),
    )
