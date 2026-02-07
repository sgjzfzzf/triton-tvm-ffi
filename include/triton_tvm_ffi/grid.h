#ifndef TRITON_TVM_FFI_GRID_H_
#define TRITON_TVM_FFI_GRID_H_

#include <cstdint>
#include <tvm/ffi/extra/cuda/base.h>
#include <tvm/ffi/tvm_ffi.h>

template <typename T>
inline tvm::ffi::Tuple<int32_t, int32_t, int32_t>
MakeGridDim(const T &grid,
            const tvm::ffi::Map<tvm::ffi::String, tvm::ffi::Any> &meta);

template <>
inline tvm::ffi::Tuple<int32_t, int32_t, int32_t>
MakeGridDim<tvm::ffi::Tuple<int32_t, int32_t, int32_t>>(
    const tvm::ffi::Tuple<int32_t, int32_t, int32_t> &grid,
    const tvm::ffi::Map<tvm::ffi::String, tvm::ffi::Any> &) {
  return grid;
}

template <>
inline tvm::ffi::Tuple<int32_t, int32_t, int32_t>
MakeGridDim<tvm::ffi::Function>(
    const tvm::ffi::Function &grid,
    const tvm::ffi::Map<tvm::ffi::String, tvm::ffi::Any> &meta) {
  tvm::ffi::Tuple<int32_t, int32_t, int32_t> tuple =
      grid(meta).cast<tvm::ffi::Tuple<int32_t, int32_t, int32_t>>();
  return MakeGridDim(tuple, meta);
}

#endif
