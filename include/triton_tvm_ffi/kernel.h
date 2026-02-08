#ifndef TRITON_TVM_FFI_KERNEL_H_
#define TRITON_TVM_FFI_KERNEL_H_

#include "macro.h"
#include <cstdint>
#include <cuda.h>
#include <unordered_map>

namespace triton_tvm_ffi {

template <const char kFnName[], const char kCubin[], size_t kSMem>
inline CUfunction GetKernel(int32_t device) {
  static std::unordered_map<int32_t, CUfunction> functions = {};
  if (functions.find(device) == functions.end()) {
    CUmodule module;
    CUfunction func;
    __CUDA_CHECK(cuModuleLoadData(&module, kCubin));
    __CUDA_CHECK(cuModuleGetFunction(&func, module, kFnName));
    if (kSMem > 49152) {
      int32_t shared_optin, shared_static;
      __CUDA_CHECK(cuDeviceGetAttribute(
          &shared_optin, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
          device));
      if (shared_optin >= kSMem) {
        __CUDA_CHECK(cuFuncGetAttribute(
            &shared_static, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, func));
        __CUDA_CHECK(cuFuncSetAttribute(
            func, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
            shared_optin - shared_static));
      }
    }
    functions[device] = func;
  }
  return functions[device];
};

} // namespace triton_tvm_ffi

#endif
