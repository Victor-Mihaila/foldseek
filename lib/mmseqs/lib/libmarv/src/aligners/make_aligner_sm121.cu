#include "score_only_alignment.cuh"
#include "score_only_alignment_interfaces.cuh"
#include "score_with_endpos_alignment.cuh"
#include "score_with_endpos_alignment_interfaces.cuh"

#include <memory>

#include "../namespace.hpp"
LIBMARV_NAMESPACE_BEGIN

#if defined(__CUDACC__) && !(__CUDACC_VER_MAJOR__ < 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ < 2))
    std::unique_ptr<MultiConfigScoreOnlyAlignerInterface> make_multiconfig_gapless_aligner_sm121_8bit(){
        struct SingleTileConfigs{
            using type = std::tuple<
                GaplessConfig<21, Approach::unused, Datatype::UInt8x4, 128, 2, 4>,
                GaplessConfig<21, Approach::unused, Datatype::UInt8x4, 128, 2, 12>,
                GaplessConfig<21, Approach::unused, Datatype::UInt8x4, 128, 2, 20>,
                GaplessConfig<21, Approach::unused, Datatype::UInt8x4, 128, 2, 28>,
                GaplessConfig<21, Approach::unused, Datatype::UInt8x4, 128, 2, 34>,
                GaplessConfig<21, Approach::unused, Datatype::UInt8x4, 512, 32, 34>,
                GaplessConfig<21, Approach::unused, Datatype::UInt8x4, 256, 16, 36>,
                GaplessConfig<21, Approach::unused, Datatype::UInt8x4, 256, 8, 38>,
                GaplessConfig<21, Approach::unused, Datatype::UInt8x4, 256, 8, 40>,
                GaplessConfig<21, Approach::unused, Datatype::UInt8x4, 256, 8, 42>,
                GaplessConfig<21, Approach::unused, Datatype::UInt8x4, 256, 8, 44>,
                GaplessConfig<21, Approach::unused, Datatype::UInt8x4, 256, 4, 46>,
                GaplessConfig<21, Approach::unused, Datatype::UInt8x4, 256, 4, 48>,
                GaplessConfig<21, Approach::unused, Datatype::UInt8x4, 256, 4, 50>,
                GaplessConfig<21, Approach::unused, Datatype::UInt8x4, 256, 4, 52>,
                GaplessConfig<21, Approach::unused, Datatype::UInt8x4, 256, 4, 54>,
                GaplessConfig<21, Approach::unused, Datatype::UInt8x4, 256, 4, 56>,
                GaplessConfig<21, Approach::unused, Datatype::UInt8x4, 256, 4, 58>,
                GaplessConfig<21, Approach::unused, Datatype::UInt8x4, 256, 2, 60>,
                GaplessConfig<21, Approach::unused, Datatype::UInt8x4, 512, 2, 62>,
                GaplessConfig<21, Approach::unused, Datatype::UInt8x4, 256, 2, 64>,
                GaplessConfig<21, Approach::unused, Datatype::UInt8x4, 512, 16, 64>
            >;

            static constexpr bool hasPerformanceHints(){
                return false;
            }
        };

        struct MultiTileConfigs{
            using type = std::tuple<
                GaplessConfig<21, Approach::unused, Datatype::UInt8x4, 512, 2, 4>,
                GaplessConfig<21, Approach::unused, Datatype::UInt8x4, 512, 2, 12>,
                GaplessConfig<21, Approach::unused, Datatype::UInt8x4, 512, 2, 20>,
                GaplessConfig<21, Approach::unused, Datatype::UInt8x4, 512, 2, 28>,
                GaplessConfig<21, Approach::unused, Datatype::UInt8x4, 512, 2, 34>,
                GaplessConfig<21, Approach::unused, Datatype::UInt8x4, 512, 32, 34>,
                GaplessConfig<21, Approach::unused, Datatype::UInt8x4, 256, 16, 36>,
                GaplessConfig<21, Approach::unused, Datatype::UInt8x4, 512, 8, 38>,
                GaplessConfig<21, Approach::unused, Datatype::UInt8x4, 256, 8, 40>,
                GaplessConfig<21, Approach::unused, Datatype::UInt8x4, 128, 8, 42>,
                GaplessConfig<21, Approach::unused, Datatype::UInt8x4, 128, 8, 44>,
                GaplessConfig<21, Approach::unused, Datatype::UInt8x4, 512, 4, 46>,
                GaplessConfig<21, Approach::unused, Datatype::UInt8x4, 512, 4, 48>,
                GaplessConfig<21, Approach::unused, Datatype::UInt8x4, 512, 4, 50>,
                GaplessConfig<21, Approach::unused, Datatype::UInt8x4, 512, 4, 52>,
                GaplessConfig<21, Approach::unused, Datatype::UInt8x4, 512, 4, 54>,
                GaplessConfig<21, Approach::unused, Datatype::UInt8x4, 512, 4, 56>,
                GaplessConfig<21, Approach::unused, Datatype::UInt8x4, 512, 4, 58>,
                GaplessConfig<21, Approach::unused, Datatype::UInt8x4, 256, 2, 60>,
                GaplessConfig<21, Approach::unused, Datatype::UInt8x4, 256, 2, 62>,
                GaplessConfig<21, Approach::unused, Datatype::UInt8x4, 256, 2, 64>,
                GaplessConfig<21, Approach::unused, Datatype::UInt8x4, 256, 16, 64>
            >;

            static constexpr bool hasPerformanceHints(){
                return false;
            }
        };

        return std::make_unique<MultiConfigScoreOnlyAligner<SingleTileConfigs, MultiTileConfigs>>(AlignmentAlgorithmE::Gapless);
    }
#endif    

    std::unique_ptr<MultiConfigScoreOnlyAlignerInterface> make_multiconfig_gapless_aligner_sm121_16bit(){
        struct SingleTileConfigs{
            using type = std::tuple<
                GaplessConfig<21, Approach::hardcodedzero, Datatype::Short2, 256, 2, 4>,
                GaplessConfig<21, Approach::hardcodedzero, Datatype::Short2, 256, 2, 12>,
                GaplessConfig<21, Approach::hardcodedzero, Datatype::Short2, 256, 2, 20>,
                GaplessConfig<21, Approach::hardcodedzero, Datatype::Short2, 256, 2, 28>,
                GaplessConfig<21, Approach::kernelparamzero, Datatype::Short2, 256, 2, 34>,
                GaplessConfig<21, Approach::kernelparamzero, Datatype::Short2, 512, 32, 34>,
                GaplessConfig<21, Approach::hardcodedzero, Datatype::Short2, 256, 16, 36>,
                GaplessConfig<21, Approach::hardcodedzero, Datatype::Short2, 256, 8, 38>,
                GaplessConfig<21, Approach::hardcodedzero, Datatype::Short2, 256, 8, 40>,
                GaplessConfig<21, Approach::hardcodedzero, Datatype::Short2, 256, 8, 42>,
                GaplessConfig<21, Approach::hardcodedzero, Datatype::Short2, 256, 8, 44>,
                GaplessConfig<21, Approach::hardcodedzero, Datatype::Short2, 256, 4, 46>,
                GaplessConfig<21, Approach::hardcodedzero, Datatype::Short2, 256, 4, 48>,
                GaplessConfig<21, Approach::hardcodedzero, Datatype::Short2, 256, 4, 50>,
                GaplessConfig<21, Approach::hardcodedzero, Datatype::Short2, 256, 4, 52>,
                GaplessConfig<21, Approach::hardcodedzero, Datatype::Short2, 256, 4, 54>,
                GaplessConfig<21, Approach::hardcodedzero, Datatype::Short2, 256, 4, 56>,
                GaplessConfig<21, Approach::hardcodedzero, Datatype::Short2, 256, 4, 58>,
                GaplessConfig<21, Approach::hardcodedzero, Datatype::Short2, 256, 2, 60>,
                GaplessConfig<21, Approach::hardcodedzero, Datatype::Short2, 256, 2, 62>,
                GaplessConfig<21, Approach::hardcodedzero, Datatype::Short2, 256, 2, 64>,
                GaplessConfig<21, Approach::hardcodedzero, Datatype::Short2, 512, 16, 64>
            >;

            static constexpr bool hasPerformanceHints(){
                return false;
            }
        };

        struct MultiTileConfigs{
            using type = std::tuple<
                GaplessConfig<21, Approach::hardcodedzero, Datatype::Short2, 512, 2, 4>,
                GaplessConfig<21, Approach::kernelparamzero, Datatype::Short2, 512, 2, 12>,
                GaplessConfig<21, Approach::hardcodedzero, Datatype::Short2, 512, 2, 20>,
                GaplessConfig<21, Approach::hardcodedzero, Datatype::Short2, 512, 2, 28>,
                GaplessConfig<21, Approach::hardcodedzero, Datatype::Short2, 512, 2, 34>,
                GaplessConfig<21, Approach::hardcodedzero, Datatype::Short2, 512, 32, 34>,
                GaplessConfig<21, Approach::hardcodedzero, Datatype::Short2, 256, 16, 36>,
                GaplessConfig<21, Approach::hardcodedzero, Datatype::Short2, 256, 8, 38>,
                GaplessConfig<21, Approach::hardcodedzero, Datatype::Short2, 512, 8, 40>,
                GaplessConfig<21, Approach::hardcodedzero, Datatype::Short2, 512, 8, 42>,
                GaplessConfig<21, Approach::hardcodedzero, Datatype::Short2, 512, 8, 44>,
                GaplessConfig<21, Approach::hardcodedzero, Datatype::Short2, 512, 4, 46>,
                GaplessConfig<21, Approach::hardcodedzero, Datatype::Short2, 512, 4, 48>,
                GaplessConfig<21, Approach::hardcodedzero, Datatype::Short2, 512, 4, 50>,
                GaplessConfig<21, Approach::hardcodedzero, Datatype::Short2, 512, 4, 52>,
                GaplessConfig<21, Approach::hardcodedzero, Datatype::Short2, 512, 4, 54>,
                GaplessConfig<21, Approach::hardcodedzero, Datatype::Short2, 512, 4, 56>,
                GaplessConfig<21, Approach::hardcodedzero, Datatype::Short2, 512, 4, 58>,
                GaplessConfig<21, Approach::hardcodedzero, Datatype::Short2, 512, 2, 60>,
                GaplessConfig<21, Approach::hardcodedzero, Datatype::Short2, 512, 2, 62>,
                GaplessConfig<21, Approach::hardcodedzero, Datatype::Short2, 512, 2, 64>,
                GaplessConfig<21, Approach::hardcodedzero, Datatype::Short2, 512, 16, 64>
            >;

            static constexpr bool hasPerformanceHints(){
                return false;
            }
        };

        return std::make_unique<MultiConfigScoreOnlyAligner<SingleTileConfigs, MultiTileConfigs>>(AlignmentAlgorithmE::Gapless);
    }

    std::unique_ptr<MultiConfigScoreOnlyAlignerInterface> make_multiconfig_gapless_aligner_sm121_8bit_if_available_else_16bit(){
                    
        #if defined(__CUDACC__) && !(__CUDACC_VER_MAJOR__ < 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ < 2))

        return make_multiconfig_gapless_aligner_sm121_8bit();

        #else

        return make_multiconfig_gapless_aligner_sm121_16bit();

        #endif
    }

LIBMARV_NAMESPACE_END