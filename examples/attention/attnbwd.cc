#include <ATen/DLConvertor.h>
#include <ATen/core/ATen_fwd.h>
#include <ATen/dlpack.h>
#include <ATen/ops/empty.h>
#include <torch/headeronly/core/DeviceType.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/extra/cuda/cubin_launcher.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/tvm_ffi.h>

#ifndef _ATTN_BWD_STUB
#define _ATTN_BWD_STUB(grid, device, stream, args, kwargs)
#endif

#ifndef _ATTN_BWD_TVM_FFI_NAME
#define _ATTN_BWD_TVM_FFI_NAME ""
#endif

tvm::ffi::Tuple<tvm::ffi::Tensor, tvm::ffi::Tensor, tvm::ffi::Tensor>
AttnBwd(tvm::ffi::Tensor q, tvm::ffi::Tensor k, tvm::ffi::Tensor v,
        const double smScale, tvm::ffi::Tensor do_, tvm::ffi::Tensor m,
        tvm::ffi::Tensor delta, const int32_t kHeadDim) {
  tvm::ffi::ShapeView qshape = q.shape(), qstride = q.strides();
  const int32_t kBatch = qshape[0], kNHead = qshape[1], kNCtx = qshape[2],
                kBlockN1 = 128;
  const double kArgKScale = smScale / log(2);
  at::Tensor qTorch = at::fromDLPack(q.ToDLPack()),
             kTorch = at::fromDLPack(k.ToDLPack()),
             vTorch = at::fromDLPack(v.ToDLPack()),
             dqTorch = at::empty_like(qTorch), dkTorch = at::empty_like(kTorch),
             dvTorch = at::empty_like(vTorch),
             argKTorch = at::mul(kTorch, kArgKScale);
  tvm::ffi::Tensor dq = tvm::ffi::Tensor::FromDLPack(at::toDLPack(dqTorch)),
                   dk = tvm::ffi::Tensor::FromDLPack(at::toDLPack(dkTorch)),
                   dv = tvm::ffi::Tensor::FromDLPack(at::toDLPack(dvTorch)),
                   argK = tvm::ffi::Tensor::FromDLPack(at::toDLPack(argKTorch));
  tvm::ffi::Tuple<int32_t, int32_t, int32_t> grid(kNCtx / kBlockN1, 1,
                                                  kBatch * kNHead);
  tvm::ffi::Array<tvm::ffi::Any> args = {
      q, argK,  v,          smScale,    do_,        dq,         dk,     dv,
      m, delta, qstride[0], qstride[1], qstride[2], qstride[3], kNHead, kNCtx};
  tvm::ffi::Map<tvm::ffi::String, tvm::ffi::Any> kwargs = {
      {"BLOCK_M1", 32}, {"BLOCK_N1", kBlockN1},  {"BLOCK_M2", 128},
      {"BLOCK_N2", 32}, {"BLK_SLICE_FACTOR", 2}, {"HEAD_DIM", kHeadDim},
      {"num_warps", 4}, {"num_stages", 5},
  };
  DLDevice device = q.device();
  void *stream = TVMFFIEnvGetStream(device.device_type, device.device_id);
  _ATTN_BWD_STUB(grid, device.device_id, stream, args, kwargs);
  return tvm::ffi::Tuple{dq, dk, dv};
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def(_ATTN_BWD_TVM_FFI_NAME, AttnBwd);
}
