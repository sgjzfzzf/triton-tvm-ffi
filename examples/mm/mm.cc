#include <ATen/DLConvertor.h>
#include <ATen/dlpack.h>
#include <tvm/ffi/extra/cuda/cubin_launcher.h>
#include <tvm/ffi/tvm_ffi.h>

#ifndef MATMUL_KERNEL_STUB
#define MATMUL_KERNEL_STUB(grid, stream, numWarps, numStages, args, kwargs)
#endif

#ifndef MATMUL_NAME
#define MATMUL_NAME ""
#endif

tvm::ffi::Tensor Matmul(tvm::ffi::Tensor a, tvm::ffi::Tensor b,
                        tvm::ffi::String activation) {
  at::Tensor atorch = at::fromDLPack(a.ToDLPack()),
             btorch = at::fromDLPack(b.ToDLPack());
  const int32_t M = atorch.size(0), K = atorch.size(1), N = btorch.size(1);
  at::Tensor ctorch = at::empty({M, N}, atorch.options());
  tvm::ffi::Function grid = tvm::ffi::Function::FromTyped(
      [M, N](const tvm::ffi::Map<tvm::ffi::String, tvm::ffi::Any> &meta)
          -> tvm::ffi::Tuple<int32_t, int32_t, int32_t> {
        const int32_t BLOCK_SIZE_M = meta["BLOCK_SIZE_M"].cast<int32_t>(),
                      BLOCK_SIZE_N = meta["BLOCK_SIZE_N"].cast<int32_t>();
        return tvm::ffi::Tuple<int32_t, int32_t, int32_t>{
            (M + BLOCK_SIZE_M - 1) / BLOCK_SIZE_M *
                ((N + BLOCK_SIZE_N - 1) / BLOCK_SIZE_N),
            1, 1};
      });
  tvm::ffi::Optional<int32_t> numWarps = std::nullopt, numStages = std::nullopt;
  DLDevice device = a.device();
  void *stream = TVMFFIEnvGetStream(device.device_type, device.device_id);
  tvm::ffi::Tensor c = tvm::ffi::Tensor::FromDLPack(at::toDLPack(ctorch));
  tvm::ffi::Array<tvm::ffi::Any> args = {a,
                                         b,
                                         c,
                                         M,
                                         N,
                                         K,
                                         atorch.stride(0),
                                         atorch.stride(1),
                                         btorch.stride(0),
                                         btorch.stride(1),
                                         ctorch.stride(0),
                                         ctorch.stride(1)};
  tvm::ffi::Map<tvm::ffi::String, tvm::ffi::Any> kwargs = {
      {"ACTIVATION", activation},
  };
  MATMUL_KERNEL_STUB(grid, stream, numWarps, numStages, args, kwargs);
  return c;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def(MATMUL_NAME, Matmul);
}
