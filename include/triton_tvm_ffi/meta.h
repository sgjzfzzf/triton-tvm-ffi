#ifndef TRITON_TVM_FFI_META_H_
#define TRITON_TVM_FFI_META_H_

#include <tvm/ffi/tvm_ffi.h>

namespace triton_tvm_ffi {

template <const char... Ks[]> struct FillMetaImpl {
  static inline void
  apply(tvm::ffi::Map<tvm::ffi::String, tvm::ffi::Any> &meta,
        tvm::ffi::Array<tvm::ffi::Any>::iterator &argsBegin,
        const tvm::ffi::Array<tvm::ffi::Any>::iterator &argsEnd,
        const tvm::ffi::Map<tvm::ffi::String, tvm::ffi::Any> &kwargs);
};

template <> struct FillMetaImpl<> {
  static inline void
  apply(tvm::ffi::Map<tvm::ffi::String, tvm::ffi::Any> &meta,
        tvm::ffi::Array<tvm::ffi::Any>::iterator &argsBegin,
        const tvm::ffi::Array<tvm::ffi::Any>::iterator &argsEnd,
        const tvm::ffi::Map<tvm::ffi::String, tvm::ffi::Any> &kwargs) {}
};

template <const char K[], const char... Ks[]> struct FillMetaImpl<K, Ks...> {
  static inline void
  apply(tvm::ffi::Map<tvm::ffi::String, tvm::ffi::Any> &meta,
        tvm::ffi::Array<tvm::ffi::Any>::iterator &argsBegin,
        const tvm::ffi::Array<tvm::ffi::Any>::iterator &argsEnd,
        const tvm::ffi::Map<tvm::ffi::String, tvm::ffi::Any> &kwargs) {
    if (argsBegin != argsEnd) {
      meta.Set(K, *argsBegin++);
    } else if (auto val = kwargs.Get(K)) {
      meta.Set(K, *val);
    }
    FillMetaImpl<Ks...>::apply(meta, argsBegin, argsEnd, kwargs);
  }
};

template <const char... Ks[]> struct FillMeta {
  static inline void
  apply(tvm::ffi::Map<tvm::ffi::String, tvm::ffi::Any> &meta,
        const tvm::ffi::Array<tvm::ffi::Any> &args,
        const tvm::ffi::Map<tvm::ffi::String, tvm::ffi::Any> &kwargs) {
    tvm::ffi::Array<tvm::ffi::Any>::iterator argsBegin = args.begin();
    tvm::ffi::Array<tvm::ffi::Any>::iterator argsEnd = args.end();
    FillMetaImpl<Ks...>::apply(meta, argsBegin, argsEnd, kwargs);
  }
};

} // namespace triton_tvm_ffi

#endif
