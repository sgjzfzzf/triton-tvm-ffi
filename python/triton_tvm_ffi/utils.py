from typing import Optional

from triton.backends.nvidia.driver import ty_to_cpp


def type_canonicalize(ty: str) -> Optional[str]:
    if ty == "constexpr":
        return None
    else:
        return ty_to_cpp(ty)
