
#ifndef MARV_CCCL_COMPAT_H
#define MARV_CCCL_COMPAT_H

#include <thrust/version.h>

#if THRUST_VERSION >= 300000

#include <cuda/barrier>

#if THRUST_VERSION < 300100

namespace thrust {
struct marv_no_init_tag {
    template <class T>
    constexpr operator T() const {
        return T();
    }
};
inline constexpr marv_no_init_tag no_init{};
}

#endif

#endif

#endif
