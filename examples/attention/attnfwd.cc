#include <ATen/DLConvertor.h>
#include <ATen/core/ATen_fwd.h>
#include <ATen/dlpack.h>
#include <ATen/ops/empty.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/extra/cuda/cubin_launcher.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/tvm_ffi.h>

#ifndef _ATTN_FWD_STUB
#define _ATTN_FWD_STUB(grid, device, stream, args, kwargs)
#endif

#ifndef _ATTN_FWD_TVM_FFI_NAME
#define _ATTN_FWD_TVM_FFI_NAME ""
#endif

tvm::ffi::Tuple<tvm::ffi::Tensor, tvm::ffi::Tensor>
AttnFwd(tvm::ffi::Tensor q, tvm::ffi::Tensor k, tvm::ffi::Tensor v, bool casual,
        float smScale) {
  const tvm::ffi::ShapeView &qshape = q.shape(), &kshape = k.shape(),
                            &vshape = v.shape();
  const int32_t kB = qshape[0], kH = qshape[1], kN = qshape[2], kQ = qshape[3],
                kK = kshape[3], kV = vshape[3], stage = casual ? 3 : 1;
  at::Tensor qTorch = at::fromDLPack(q.ToDLPack()),
             oTorch = at::empty_like(qTorch),
             mTorch =
                 at::empty({kB, kH, kN}, qTorch.options().dtype(at::kFloat));
  tvm::ffi::Tensor o = tvm::ffi::Tensor::FromDLPack(at::toDLPack(oTorch)),
                   m = tvm::ffi::Tensor::FromDLPack(at::toDLPack(mTorch));
  tvm::ffi::Function grid = tvm::ffi::Function::FromTyped(
      [kB, kH, kN](const tvm::ffi::Map<tvm::ffi::String, tvm::ffi::Any> &meta)
          -> tvm::ffi::Tuple<int32_t, int32_t, int32_t> {
        const int32_t kBlockM = meta["BLOCK_M"].cast<int32_t>();
        return tvm::ffi::Tuple<int32_t, int32_t, int32_t>(
            (kN + kBlockM - 1) / kBlockM, kB * kH, 1);
      });
  tvm::ffi::Array<tvm::ffi::Any> args = {smScale, m, kB, kH, q, k, v, o, kN};
  tvm::ffi::Map<tvm::ffi::String, tvm::ffi::Any> kwargs = {
      {"HEAD_DIM", kK},
      {"STAGE", stage},
  };
  DLDevice device = q.device();
  void *stream = TVMFFIEnvGetStream(device.device_type, device.device_id);
  _ATTN_FWD_STUB(grid, device.device_id, stream, args, kwargs);
  return tvm::ffi::Tuple{m, o};
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def(_ATTN_FWD_TVM_FFI_NAME, AttnFwd);
}
