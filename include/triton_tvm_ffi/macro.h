#ifndef TRITON_TVM_FFI_MACRO_H_
#define TRITON_TVM_FFI_MACRO_H_

#include <cuda.h>
#include <sstream>
#include <stdexcept>
#include <string>

#define __CUDA_CHECK(__code)                                                   \
  do {                                                                         \
    if ((__code) != CUDA_SUCCESS) {                                            \
      const char *errorName = nullptr, *errorStr = nullptr;                    \
      cuGetErrorName((__code), &errorName);                                    \
      cuGetErrorString((__code), &errorStr);                                   \
      std::ostringstream __oss;                                                \
      __oss << "[" << errorName << "] " << errorStr << ", at " << __FILE__     \
            << ":" << __LINE__;                                                \
      throw std::runtime_error(__oss.str());                                   \
    }                                                                          \
  } while (false)

#endif
