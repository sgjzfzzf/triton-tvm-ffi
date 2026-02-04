#include <ATen/DLConvertor.h>
#include <ATen/dlpack.h>
#include <tvm/ffi/extra/cuda/cubin_launcher.h>
#include <tvm/ffi/tvm_ffi.h>

#ifndef SOFTMAX_KERNEL_STUB
#define SOFTMAX_KERNEL_STUB(grid, stream, numWarps, numStages, output, input,  \
                            inputStride, outputStride, nRows, nCols,           \
                            BLOCK_SIZE)
#endif

#ifndef SOFTMAX_NAME
#define SOFTMAX_NAME ""
#endif

tvm::ffi::Tensor Softmax(tvm::ffi::Tensor x) {
  at::Tensor xtorch = at::fromDLPack(x.ToDLPack());
  at::Tensor ytorch = at::empty_like(xtorch);
  uint32_t nRows = xtorch.size(0), nCols = xtorch.size(1), numWarps = 8,
           numStages = 4, xStride = xtorch.stride(0),
           yStride = ytorch.stride(0),
           BLOCK_SIZE = 1u << (32 - __builtin_clz(nCols - 1));
  tvm::ffi::Tensor y = tvm::ffi::Tensor::FromDLPack(at::toDLPack(ytorch));
  tvm::ffi::Tuple<int32_t, int32_t, int32_t> grid{nRows / 1024, 1, 1};
  DLDevice device = x.device();
  void *stream = TVMFFIEnvGetStream(device.device_type, device.device_id);
  SOFTMAX_KERNEL_STUB(grid, stream, numWarps, numStages, y, x, xStride, yStride,
                      nRows, nCols, BLOCK_SIZE);
  return y;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def(SOFTMAX_NAME, Softmax);
}
