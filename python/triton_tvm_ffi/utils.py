import importlib.resources
from importlib.resources.abc import Traversable
from typing import List, Optional

from triton.backends.nvidia.driver import ty_to_cpp


def include_paths() -> List[str]:
    pkg_path: Traversable = importlib.resources.files("triton_tvm_ffi")
    return [str(pkg_path / "include"), str(pkg_path / "../../include")]


def type_canonicalize(ty: str) -> Optional[str]:
    if ty == "constexpr":
        return None
    else:
        return ty_to_cpp(ty)
