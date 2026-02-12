#include <ATen/DLConvertor.h>
#include <ATen/core/ATen_fwd.h>
#include <ATen/dlpack.h>
#include <ATen/ops/empty.h>
#include <torch/headeronly/core/DeviceType.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/extra/cuda/cubin_launcher.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/tvm_ffi.h>

#ifndef _ATTN_BWD_PREPROCESS_STUB
#define _ATTN_BWD_PREPROCESS_STUB(grid, device, stream, args, kwargs)
#endif

#ifndef _ATTN_BWD_PREPROCESS_TVM_FFI_NAME
#define _ATTN_BWD_PREPROCESS_TVM_FFI_NAME ""
#endif

tvm::ffi::Tensor AttnBwdPreprocess(tvm::ffi::Tensor o, tvm::ffi::Tensor do_,
                                   tvm::ffi::Shape mshape,
                                   const int32_t kHeadDim) {
  const int32_t kBatch = mshape[0], kNHead = mshape[1], kNCtx = mshape[2],
                kPreBlock = 128;
  at::Tensor deltaTorch = at::empty(mshape, at::kFloat, std::nullopt,
                                    at::Device(at::kCUDA, o.device().device_id),
                                    std::nullopt, std::nullopt);
  tvm::ffi::Tensor delta =
      tvm::ffi::Tensor::FromDLPack(at::toDLPack(deltaTorch));
  tvm::ffi::Tuple<int32_t, int32_t, int32_t> grid(kNCtx / kPreBlock,
                                                  kBatch * kNHead, 1);
  tvm::ffi::Array<tvm::ffi::Any> args = {o, do_, delta, kBatch, kNHead};
  tvm::ffi::Map<tvm::ffi::String, tvm::ffi::Any> kwargs = {
      {"N_CTX", kNCtx},
      {"BLOCK_M", kPreBlock},
      {"HEAD_DIM", kHeadDim},
  };
  DLDevice device = o.device();
  void *stream = TVMFFIEnvGetStream(device.device_type, device.device_id);
  _ATTN_BWD_PREPROCESS_STUB(grid, device.device_id, stream, args, kwargs);
  return delta;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def(_ATTN_BWD_PREPROCESS_TVM_FFI_NAME, AttnBwdPreprocess);
}
