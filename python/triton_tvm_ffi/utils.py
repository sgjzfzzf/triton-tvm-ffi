import sysconfig
from typing import List, Optional

from triton.backends.nvidia.driver import ty_to_cpp


def include_paths() -> List[str]:
    pkg_path: str = sysconfig.get_path("purelib")
    return [f"{pkg_path}/triton_tvm_ffi/include"]


def type_canonicalize(ty: str) -> Optional[str]:
    if ty == "constexpr":
        return None
    else:
        return ty_to_cpp(ty)
