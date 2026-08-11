// CCCL 3 / CUDA 13 compatibility shim for the vendored libmarv.
//
// libmarv is kept byte-identical to its upstream tree, and upstream targets CUDA 12
// (CCCL 2.x) -- see its Readme. Two things it relies on do not survive CCCL 3.0, which
// ships with CUDA 13. Rather than patch libmarv, this header is force-included (nvcc
// --pre-include) into every libmarv CUDA translation unit from lib/mmseqs/CMakeLists.txt.
//
//  1. `vec.resize(n, thrust::no_init)` -- Thrust removed the uninitialized-resize tag.
//     Re-supplied below as a value convertible to any element type, so those calls bind to
//     the `resize(n, const value_type&)` overload instead. Consequence: the affected temp
//     buffers are value-initialized rather than left uninitialized -- correct, just a
//     redundant fill on allocation.
//
//  2. `cuda::aligned_size_t` -- still exists, but no longer arrives transitively via
//     <cooperative_groups/memcpy_async.h>. Pulled in explicitly below.
//
// Delete this file and its wiring in lib/mmseqs/CMakeLists.txt once libmarv itself builds
// against CCCL 3 (upstream fix: plain `resize(n)` or cuda::no_init, plus the include).

#ifndef MARV_CCCL_COMPAT_H
#define MARV_CCCL_COMPAT_H

#include <thrust/version.h>

#if THRUST_VERSION >= 300000

// (2) provides cuda::aligned_size_t
#include <cuda/barrier>

// (1) stand-in for the removed thrust::no_init tag.
//     CCCL 3.0/3.1 (CUDA 13.0/13.1) dropped it; CCCL 3.2 (CUDA 13.2) reinstated it as
//     thrust::no_init_t in <thrust/detail/vector_base.h>. Defining ours on 3.2+ would be a
//     redefinition, so bound the workaround to the versions that actually need it.
#if THRUST_VERSION < 300200

namespace thrust {
struct marv_no_init_tag {
    template <class T>
    constexpr operator T() const {
        return T();
    }
};
inline constexpr marv_no_init_tag no_init{};
}

#endif // THRUST_VERSION < 300200

#endif // THRUST_VERSION >= 300000

#endif
