#include <ATen/DLConvertor.h>
#include <ATen/dlpack.h>
#include <tvm/ffi/extra/cuda/cubin_launcher.h>
#include <tvm/ffi/tvm_ffi.h>

#ifndef ADD_KERNEL_STUB
#define ADD_KERNEL_STUB(grid, stream, numWarps, numStages, x, y, output,       \
                        numel, BLOCK_SIZE)
#endif

#ifndef ADD_NAME
#define ADD_NAME ""
#endif

tvm::ffi::Tensor Add(tvm::ffi::Tensor x, tvm::ffi::Tensor y) {
  at::Tensor xtorch = at::fromDLPack(x.ToDLPack());
  at::Tensor otorch = at::empty_like(xtorch);
  int64_t numel = otorch.numel();
  tvm::ffi::Tensor output = tvm::ffi::Tensor::FromDLPack(at::toDLPack(otorch));
  tvm::ffi::Tuple<int32_t, int32_t, int32_t> grid{(numel + 1023) / 1024, 1, 1};
  // TODO: check the performance loss after enabling `Optional`
  tvm::ffi::Optional<int32_t> numWarps = std::nullopt, numStages = std::nullopt;
  DLDevice device = x.device();
  void *stream = TVMFFIEnvGetStream(device.device_type, device.device_id);
  ADD_KERNEL_STUB(grid, stream, numWarps, numStages, x, y, output, numel, 1024);
  return output;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def(ADD_NAME, Add);
}
