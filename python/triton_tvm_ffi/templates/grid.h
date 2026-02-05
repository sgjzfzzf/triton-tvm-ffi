#ifndef TRITON_TVM_FFI_GRID_H
#define TRITON_TVM_FFI_GRID_H

#include <cstdint>
#include <tvm/ffi/extra/cuda/base.h>
#include <tvm/ffi/tvm_ffi.h>

template <typename T>
inline tvm::ffi::dim3
MakeGridDim(const T &grid,
            const tvm::ffi::Map<tvm::ffi::String, tvm::ffi::Any> &meta);

template <>
inline tvm::ffi::dim3 MakeGridDim<tvm::ffi::Tuple<int32_t, int32_t, int32_t>>(
    const tvm::ffi::Tuple<int32_t, int32_t, int32_t> &grid,
    const tvm::ffi::Map<tvm::ffi::String, tvm::ffi::Any> &) {
  return tvm::ffi::dim3(grid.get<0>(), grid.get<1>(), grid.get<2>());
}

template <>
inline tvm::ffi::dim3 MakeGridDim<tvm::ffi::Function>(
    const tvm::ffi::Function &grid,
    const tvm::ffi::Map<tvm::ffi::String, tvm::ffi::Any> &meta) {
  tvm::ffi::Tuple<int32_t, int32_t, int32_t> tuple =
      grid(meta).cast<tvm::ffi::Tuple<int32_t, int32_t, int32_t>>();
  return MakeGridDim(tuple, meta);
}

#endif
