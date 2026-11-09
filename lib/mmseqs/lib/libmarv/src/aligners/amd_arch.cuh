#ifndef MARV_AMD_ARCH_CUH
#define MARV_AMD_ARCH_CUH

#if defined(__HIP_PLATFORM_AMD__)

#include <cstring>

#include "../cuda_errorcheck.cuh"

#include "../namespace.hpp"
LIBMARV_NAMESPACE_BEGIN

// gfx900 and gfx90a both report compute capability 9.0, so tell them apart by gcnArchName ("gfx90a:sramecc+:xnack-")
inline bool isAmdGfx90a(int deviceId){
    hipDeviceProp_t prop;
    CUDACHECK(hipGetDeviceProperties(&prop, deviceId));
    return std::strncmp(prop.gcnArchName, "gfx90a", 6) == 0;
}

LIBMARV_NAMESPACE_END

#endif

#endif
