from functools import cached_property
from io import TextIOWrapper
from pathlib import Path
from typing import Any, Callable, Final, List, Optional, Sequence, Union

import jinja2
import torch.utils.cpp_extension
import tvm_ffi

from .jit import TVMFFIJITFunction
from .utils import include_paths


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
        self.env: Final[jinja2.Environment] = jinja2.Environment(
            loader=jinja2.PackageLoader("triton_tvm_ffi", "templates"),
            trim_blocks=True,
        )
        self.tpl: Final[jinja2.Template] = self.env.get_template("gendef.cc.j2")

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
        return self.tpl.render(
            code=self.code, fns=self.fns, name=self.name, uniquename=self.uniquename
        )

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
            return tvm_ffi.get_global_func(self.uniquename)


def wrap(
    fns: List[TVMFFIJITFunction],
    code: Union[str, Path, TextIOWrapper],
    extra_cflags: Optional[Sequence[str]] = None,
    extra_cuda_cflags: Optional[Sequence[str]] = None,
    extra_ldflags: Optional[Sequence[str]] = None,
    extra_include_paths: Optional[Sequence[Union[str, Path]]] = None,
) -> TVMFFIWrapperFunction:
    def decorate(fn: Union[str, Callable[..., Any]]) -> TVMFFIWrapperFunction:
        return TVMFFIWrapperFunction(
            fn if isinstance(fn, str) else fn.__name__,
            fns,
            code,
            extra_cflags,
            extra_cuda_cflags,
            extra_ldflags,
            include_paths() + (extra_include_paths or []),
        )

    return decorate


def torch_wrap(
    fns: List[TVMFFIJITFunction],
    code: Union[str, Path, TextIOWrapper],
    extra_cflags: Optional[Sequence[str]] = None,
    extra_cuda_cflags: Optional[Sequence[str]] = None,
    extra_ldflags: Optional[Sequence[str]] = None,
    extra_include_paths: Optional[Sequence[Union[str, Path]]] = None,
) -> TVMFFIWrapperFunction:
    cuda_home: str = tvm_ffi.cpp.extension._find_cuda_home()
    return wrap(
        fns,
        code,
        extra_ldflags=[
            "-Wl,--no-as-needed",
            f"-L{cuda_home}/lib64",
            *map(
                lambda path: f"-L{path}",
                torch.utils.cpp_extension.library_paths(),
            ),
            "-lcuda",
            "-lc10",
            "-ltorch",
        ]
        + (extra_ldflags or []),
        extra_cflags=extra_cflags,
        extra_cuda_cflags=extra_cuda_cflags,
        extra_include_paths=[
            f"{cuda_home}/include",
            *torch.utils.cpp_extension.include_paths(),
        ]
        + (extra_include_paths or []),
    )
