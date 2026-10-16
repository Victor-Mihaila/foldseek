#include "score_only_alignment_interfaces.cuh"

#include <memory>


#include "make_aligner_headers.cuh"
#include "../cuda_errorcheck.cuh"

#include "../namespace.hpp"
LIBMARV_NAMESPACE_BEGIN

//Construct aligner with at least 8-bit computations. Could use higher precision if 8-bit is unavailable
std::unique_ptr<MultiConfigScoreOnlyAlignerInterface> make_multiconfig_gapless_aligner_8bit(){
    int deviceId = 0;
    CUDACHECK(cudaGetDevice(&deviceId));
    int ccMajor = 0;
    int ccMinor = 0;
    CUDACHECK(cudaDeviceGetAttribute(&ccMajor, cudaDevAttrComputeCapabilityMajor, deviceId));
    CUDACHECK(cudaDeviceGetAttribute(&ccMinor, cudaDevAttrComputeCapabilityMinor, deviceId));


#if defined(__CUDACC__)

#ifdef MARV_HAVE_SM75
    if(ccMajor == 7 && ccMinor == 5){
        return make_multiconfig_gapless_aligner_sm75_16bit();
    }
#endif
#ifdef MARV_HAVE_SM80
    if(ccMajor == 8 && ccMinor == 0){
        return make_multiconfig_gapless_aligner_sm80_16bit();
    }
#endif
#ifdef MARV_HAVE_SM86
    if(ccMajor == 8 && ccMinor == 6){
        return make_multiconfig_gapless_aligner_sm86_16bit();
    }
#endif
#ifdef MARV_HAVE_SM87
    if(ccMajor == 8 && ccMinor == 7){
        return make_multiconfig_gapless_aligner_sm87_16bit();
    }
#endif
#ifdef MARV_HAVE_SM89
    if(ccMajor == 8 && ccMinor == 9){
        return make_multiconfig_gapless_aligner_sm89_16bit();
    }
#endif
#ifdef MARV_HAVE_SM90
    if(ccMajor == 9 && ccMinor == 0){
        return make_multiconfig_gapless_aligner_sm90_16bit();
    }
#endif
#ifdef MARV_HAVE_SM100
    if(ccMajor == 10 && ccMinor == 0){
        return make_multiconfig_gapless_aligner_sm100_16bit();
    }
#endif
#ifdef MARV_HAVE_SM103
    if(ccMajor == 10 && ccMinor == 3){
        return make_multiconfig_gapless_aligner_sm103_16bit();
    }
#endif
#ifdef MARV_HAVE_SM110
    if((ccMajor == 11 && ccMinor == 0) || (ccMajor == 10 && ccMinor == 1)){
        return make_multiconfig_gapless_aligner_sm110_16bit();
    }
#endif
#ifdef MARV_HAVE_SM120
    if(ccMajor == 12 && ccMinor == 0){
        return make_multiconfig_gapless_aligner_sm120_8bit_if_available_else_16bit();
    }
#endif
#ifdef MARV_HAVE_SM121
    if(ccMajor == 12 && ccMinor == 1){
        return make_multiconfig_gapless_aligner_sm121_8bit_if_available_else_16bit();
    }
#endif
    return make_multiconfig_gapless_aligner_default_ptx_16bit();

#else
    if(ccMajor == 9 && ccMinor == 0){
        return make_multiconfig_gapless_aligner_amd_gfx90a();
    }else if(ccMajor == 9 && ccMinor == 4){
        return make_multiconfig_gapless_aligner_amd_gfx942();
    }else if(ccMajor == 11 && ccMinor == 5){
        return make_multiconfig_gapless_aligner_amd_gfx11();
    }else{
        return make_multiconfig_gapless_aligner_amd_default();
    }
#endif
}

//Construct aligner with at least 16-bit computations. Could use higher precision if 16-bit is unavailable
std::unique_ptr<MultiConfigScoreOnlyAlignerInterface> make_multiconfig_gapless_aligner_16bit(){
    int deviceId = 0;
    CUDACHECK(cudaGetDevice(&deviceId));
    int ccMajor = 0;
    int ccMinor = 0;
    CUDACHECK(cudaDeviceGetAttribute(&ccMajor, cudaDevAttrComputeCapabilityMajor, deviceId));
    CUDACHECK(cudaDeviceGetAttribute(&ccMinor, cudaDevAttrComputeCapabilityMinor, deviceId));


#if defined(__CUDACC__)

#ifdef MARV_HAVE_SM75
    if(ccMajor == 7 && ccMinor == 5){
        return make_multiconfig_gapless_aligner_sm75_16bit();
    }
#endif
#ifdef MARV_HAVE_SM80
    if(ccMajor == 8 && ccMinor == 0){
        return make_multiconfig_gapless_aligner_sm80_16bit();
    }
#endif
#ifdef MARV_HAVE_SM86
    if(ccMajor == 8 && ccMinor == 6){
        return make_multiconfig_gapless_aligner_sm86_16bit();
    }
#endif
#ifdef MARV_HAVE_SM87
    if(ccMajor == 8 && ccMinor == 7){
        return make_multiconfig_gapless_aligner_sm87_16bit();
    }
#endif
#ifdef MARV_HAVE_SM89
    if(ccMajor == 8 && ccMinor == 9){
        return make_multiconfig_gapless_aligner_sm89_16bit();
    }
#endif
#ifdef MARV_HAVE_SM90
    if(ccMajor == 9 && ccMinor == 0){
        return make_multiconfig_gapless_aligner_sm90_16bit();
    }
#endif
#ifdef MARV_HAVE_SM100
    if(ccMajor == 10 && ccMinor == 0){
        return make_multiconfig_gapless_aligner_sm100_16bit();
    }
#endif
#ifdef MARV_HAVE_SM103
    if(ccMajor == 10 && ccMinor == 3){
        return make_multiconfig_gapless_aligner_sm103_16bit();
    }
#endif
#ifdef MARV_HAVE_SM110
    if((ccMajor == 11 && ccMinor == 0) || (ccMajor == 10 && ccMinor == 1)){
        return make_multiconfig_gapless_aligner_sm110_16bit();
    }
#endif
#ifdef MARV_HAVE_SM120
    if(ccMajor == 12 && ccMinor == 0){
        return make_multiconfig_gapless_aligner_sm120_16bit();
    }
#endif
#ifdef MARV_HAVE_SM121
    if(ccMajor == 12 && ccMinor == 1){
        return make_multiconfig_gapless_aligner_sm121_16bit();
    }
#endif
    return make_multiconfig_gapless_aligner_default_ptx_16bit();

#else
    if(ccMajor == 9 && ccMinor == 0){
        return make_multiconfig_gapless_aligner_amd_gfx90a();
    }else if(ccMajor == 9 && ccMinor == 4){
        return make_multiconfig_gapless_aligner_amd_gfx942();
    }else if(ccMajor == 11 && ccMinor == 5){
        return make_multiconfig_gapless_aligner_amd_gfx11();
    }else{
        return make_multiconfig_gapless_aligner_amd_default();
    }
#endif
}

//Construct aligner with 32-bit computations.
std::unique_ptr<MultiConfigScoreOnlyAlignerInterface> make_multiconfig_gapless_aligner_32bit(){
    int deviceId = 0;
    CUDACHECK(cudaGetDevice(&deviceId));
    int ccMajor = 0;
    int ccMinor = 0;
    CUDACHECK(cudaDeviceGetAttribute(&ccMajor, cudaDevAttrComputeCapabilityMajor, deviceId));
    CUDACHECK(cudaDeviceGetAttribute(&ccMinor, cudaDevAttrComputeCapabilityMinor, deviceId));


#if defined(__CUDACC__)

#ifdef MARV_HAVE_SM75
    if(ccMajor == 7 && ccMinor == 5){
        return make_multiconfig_gapless_aligner_sm75_32bit();
    }
#endif
    return make_multiconfig_gapless_aligner_default_ptx_32bit();

#else
    return make_multiconfig_gapless_aligner_default_ptx_32bit();
#endif
}





std::unique_ptr<MultiConfigScoreOnlyAlignerInterface> make_multiconfig_sw_aligner_32bit(){
    int deviceId = 0;
    CUDACHECK(cudaGetDevice(&deviceId));
    int ccMajor = 0;
    int ccMinor = 0;
    CUDACHECK(cudaDeviceGetAttribute(&ccMajor, cudaDevAttrComputeCapabilityMajor, deviceId));
    CUDACHECK(cudaDeviceGetAttribute(&ccMinor, cudaDevAttrComputeCapabilityMinor, deviceId));


    return make_multiconfig_sw_aligner_default_ptx_32bit(); 
}


LIBMARV_NAMESPACE_END