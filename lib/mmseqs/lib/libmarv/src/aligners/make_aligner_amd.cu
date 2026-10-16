#include "score_only_alignment.cuh"
#include "score_only_alignment_interfaces.cuh"

#include <memory>

#include "../namespace.hpp"
LIBMARV_NAMESPACE_BEGIN

    // AMD/HIP gapless aligner. Uses 32-bit float configs because RDNA has no
    // DPX packed-int16 instructions (Short2) and fp16 (Half2) loses precision
    // for large scores (e.g. foldseek 3di self-alignments).
    std::unique_ptr<MultiConfigScoreOnlyAlignerInterface> make_multiconfig_gapless_aligner_amd(){
        struct SingleTileConfigs{
            using type = std::tuple<
                GaplessConfig<21,Approach::unused, Datatype::Float,128,4,16>,
                GaplessConfig<21,Approach::unused, Datatype::Float,128,4,32>,
                GaplessConfig<21,Approach::unused, Datatype::Float,128,8,32>,
                GaplessConfig<21,Approach::unused, Datatype::Float,128,16,32>
            >;

            static constexpr bool hasPerformanceHints(){
                return false;
            }
        };

        using MultiTileConfigs = SingleTileConfigs;


        return std::make_unique<MultiConfigScoreOnlyAligner<SingleTileConfigs, MultiTileConfigs>>(AlignmentAlgorithmE::Gapless);
    }


    std::unique_ptr<MultiConfigScoreOnlyAlignerInterface> make_multiconfig_gapless_aligner_amd_gfx90a(){
        struct SingleTileConfigs{
            using type = std::tuple<
                GaplessConfig<21, Approach::kernelparamzero, Datatype::Half2, 512, 4, 4>,
                GaplessConfig<21, Approach::kernelparamzero, Datatype::Half2, 512, 4, 8>,
                GaplessConfig<21, Approach::kernelparamzero, Datatype::Half2, 512, 4, 12>,
                GaplessConfig<21, Approach::kernelparamzero, Datatype::Half2, 512, 4, 16>,
                GaplessConfig<21, Approach::hardcodedzero, Datatype::Half2, 512, 4, 20>,
                GaplessConfig<21, Approach::kernelparamzero, Datatype::Half2, 512, 4, 24>,
                GaplessConfig<21, Approach::hardcodedzero, Datatype::Half2, 512, 4, 28>,
                GaplessConfig<21, Approach::hardcodedzero, Datatype::Half2, 512, 8, 16>,
                GaplessConfig<21, Approach::hardcodedzero, Datatype::Half2, 512, 4, 36>,
                GaplessConfig<21, Approach::hardcodedzero, Datatype::Half2, 512, 4, 40>,
                GaplessConfig<21, Approach::hardcodedzero, Datatype::Half2, 512, 4, 44>,
                GaplessConfig<21, Approach::kernelparamzero, Datatype::Half2, 512, 8, 24>,
                GaplessConfig<21, Approach::hardcodedzero, Datatype::Half2, 512, 4, 52>,
                GaplessConfig<21, Approach::kernelparamzero, Datatype::Half2, 512, 8, 28>,
                GaplessConfig<21, Approach::kernelparamzero, Datatype::Half2, 512, 4, 60>,
                GaplessConfig<21, Approach::hardcodedzero, Datatype::Half2, 512, 8, 32>,
                GaplessConfig<21, Approach::kernelparamzero, Datatype::Half2, 512, 8, 36>,
                GaplessConfig<21, Approach::kernelparamzero, Datatype::Half2, 512, 8, 40>,
                GaplessConfig<21, Approach::kernelparamzero, Datatype::Half2, 512, 8, 44>,
                GaplessConfig<21, Approach::kernelparamzero, Datatype::Half2, 512, 8, 48>,
                GaplessConfig<21, Approach::kernelparamzero, Datatype::Half2, 512, 8, 52>,
                GaplessConfig<21, Approach::kernelparamzero, Datatype::Half2, 512, 8, 56>,
                GaplessConfig<21, Approach::hardcodedzero, Datatype::Half2, 512, 8, 60>,
                GaplessConfig<21, Approach::hardcodedzero, Datatype::Half2, 512, 8, 64>,
                GaplessConfig<21, Approach::hardcodedzero, Datatype::Half2, 512, 16, 36>,
                GaplessConfig<21, Approach::hardcodedzero, Datatype::Half2, 512, 16, 40>,
                GaplessConfig<21, Approach::kernelparamzero, Datatype::Half2, 512, 16, 44>,
                GaplessConfig<21, Approach::kernelparamzero, Datatype::Half2, 512, 16, 48>
            >;

            static constexpr bool hasPerformanceHints(){
                return false;
            }
        };

        using MultiTileConfigs = SingleTileConfigs;


        return std::make_unique<MultiConfigScoreOnlyAligner<SingleTileConfigs, MultiTileConfigs>>(AlignmentAlgorithmE::Gapless);
    }

    std::unique_ptr<MultiConfigScoreOnlyAlignerInterface> make_multiconfig_gapless_aligner_amd_gfx942(){
        struct SingleTileConfigs{
            using type = std::tuple<
                GaplessConfig<21, Approach::kernelparamzero, Datatype::Half2, 512, 4, 4>,
                GaplessConfig<21, Approach::kernelparamzero, Datatype::Half2, 512, 4, 8>,
                GaplessConfig<21, Approach::kernelparamzero, Datatype::Half2, 512, 4, 12>,
                GaplessConfig<21, Approach::kernelparamzero, Datatype::Half2, 512, 4, 16>,
                GaplessConfig<21, Approach::kernelparamzero, Datatype::Half2, 512, 4, 20>,
                GaplessConfig<21, Approach::kernelparamzero, Datatype::Half2, 512, 4, 24>,
                GaplessConfig<21, Approach::kernelparamzero, Datatype::Half2, 512, 4, 28>,
                GaplessConfig<21, Approach::kernelparamzero, Datatype::Half2, 512, 4, 32>,
                GaplessConfig<21, Approach::kernelparamzero, Datatype::Half2, 512, 4, 36>,
                GaplessConfig<21, Approach::kernelparamzero, Datatype::Half2, 512, 8, 20>,
                GaplessConfig<21, Approach::kernelparamzero, Datatype::Half2, 512, 4, 44>,
                GaplessConfig<21, Approach::kernelparamzero, Datatype::Half2, 512, 8, 24>,
                GaplessConfig<21, Approach::kernelparamzero, Datatype::Half2, 512, 8, 28>,
                GaplessConfig<21, Approach::kernelparamzero, Datatype::Half2, 512, 8, 32>,
                GaplessConfig<21, Approach::kernelparamzero, Datatype::Half2, 512, 8, 36>,
                GaplessConfig<21, Approach::kernelparamzero, Datatype::Half2, 512, 8, 40>,
                GaplessConfig<21, Approach::kernelparamzero, Datatype::Half2, 512, 8, 44>,
                GaplessConfig<21, Approach::kernelparamzero, Datatype::Half2, 512, 8, 48>,
                GaplessConfig<21, Approach::kernelparamzero, Datatype::Half2, 512, 8, 52>,
                GaplessConfig<21, Approach::kernelparamzero, Datatype::Half2, 512, 8, 56>,
                GaplessConfig<21, Approach::kernelparamzero, Datatype::Half2, 512, 8, 60>,
                GaplessConfig<21, Approach::kernelparamzero, Datatype::Half2, 512, 8, 64>,
                GaplessConfig<21, Approach::kernelparamzero, Datatype::Half2, 512, 16, 36>,
                GaplessConfig<21, Approach::kernelparamzero, Datatype::Half2, 512, 16, 40>,
                GaplessConfig<21, Approach::kernelparamzero, Datatype::Half2, 512, 16, 44>,
                GaplessConfig<21, Approach::kernelparamzero, Datatype::Half2, 512, 16, 48>
            >;

            static constexpr bool hasPerformanceHints(){
                return false;
            }
        };

        using MultiTileConfigs = SingleTileConfigs;


        return std::make_unique<MultiConfigScoreOnlyAligner<SingleTileConfigs, MultiTileConfigs>>(AlignmentAlgorithmE::Gapless);
    }

    std::unique_ptr<MultiConfigScoreOnlyAlignerInterface> make_multiconfig_gapless_aligner_amd_gfx11(){
        struct SingleTileConfigs{
            using type = std::tuple<
                GaplessConfig<21, Approach::kernelparamzero, Datatype::Half2, 256, 2, 4>,
                GaplessConfig<21, Approach::hardcodedzero, Datatype::Half2, 512, 2, 8>,
                GaplessConfig<21, Approach::kernelparamzero, Datatype::Half2, 512, 2, 12>,
                GaplessConfig<21, Approach::hardcodedzero, Datatype::Half2, 512, 2, 16>,
                GaplessConfig<21, Approach::hardcodedzero, Datatype::Half2, 512, 2, 20>,
                GaplessConfig<21, Approach::hardcodedzero, Datatype::Half2, 512, 2, 24>,
                GaplessConfig<21, Approach::hardcodedzero, Datatype::Half2, 512, 2, 28>,
                GaplessConfig<21, Approach::hardcodedzero, Datatype::Half2, 512, 2, 32>,
                GaplessConfig<21, Approach::hardcodedzero, Datatype::Half2, 512, 2, 36>,
                GaplessConfig<21, Approach::hardcodedzero, Datatype::Half2, 512, 2, 40>,
                GaplessConfig<21, Approach::kernelparamzero, Datatype::Half2, 512, 2, 44>,
                GaplessConfig<21, Approach::kernelparamzero, Datatype::Half2, 512, 4, 24>,
                GaplessConfig<21, Approach::kernelparamzero, Datatype::Half2, 512, 2, 52>,
                GaplessConfig<21, Approach::kernelparamzero, Datatype::Half2, 512, 4, 28>,
                GaplessConfig<21, Approach::kernelparamzero, Datatype::Half2, 512, 2, 60>,
                GaplessConfig<21, Approach::kernelparamzero, Datatype::Half2, 512, 4, 32>,
                GaplessConfig<21, Approach::kernelparamzero, Datatype::Half2, 512, 4, 36>,
                GaplessConfig<21, Approach::hardcodedzero, Datatype::Half2, 512, 4, 40>,
                GaplessConfig<21, Approach::kernelparamzero, Datatype::Half2, 512, 4, 44>,
                GaplessConfig<21, Approach::hardcodedzero, Datatype::Half2, 512, 8, 24>,
                GaplessConfig<21, Approach::kernelparamzero, Datatype::Half2, 512, 4, 52>,
                GaplessConfig<21, Approach::hardcodedzero, Datatype::Half2, 512, 8, 28>,
                GaplessConfig<21, Approach::kernelparamzero, Datatype::Half2, 512, 4, 60>,
                GaplessConfig<21, Approach::kernelparamzero, Datatype::Half2, 512, 8, 32>,
                GaplessConfig<21, Approach::hardcodedzero, Datatype::Half2, 512, 8, 36>,
                GaplessConfig<21, Approach::hardcodedzero, Datatype::Half2, 512, 8, 40>,
                GaplessConfig<21, Approach::kernelparamzero, Datatype::Half2, 512, 8, 44>,
                GaplessConfig<21, Approach::hardcodedzero, Datatype::Half2, 512, 16, 24>,
                GaplessConfig<21, Approach::hardcodedzero, Datatype::Half2, 512, 8, 52>,
                GaplessConfig<21, Approach::hardcodedzero, Datatype::Half2, 512, 16, 28>,
                GaplessConfig<21, Approach::kernelparamzero, Datatype::Half2, 512, 8, 60>,
                GaplessConfig<21, Approach::hardcodedzero, Datatype::Half2, 512, 8, 64>,
                GaplessConfig<21, Approach::hardcodedzero, Datatype::Half2, 512, 16, 36>,
                GaplessConfig<21, Approach::hardcodedzero, Datatype::Half2, 512, 16, 40>,
                GaplessConfig<21, Approach::hardcodedzero, Datatype::Half2, 512, 16, 44>,
                GaplessConfig<21, Approach::kernelparamzero, Datatype::Half2, 512, 16, 48>,
                GaplessConfig<21, Approach::half2fmarelu, Datatype::Half2, 512, 16, 52>,
                GaplessConfig<21, Approach::half2fmarelu, Datatype::Half2, 512, 16, 56>,
                GaplessConfig<21, Approach::half2fmarelu, Datatype::Half2, 512, 16, 60>,
                GaplessConfig<21, Approach::half2fmarelu, Datatype::Half2, 512, 16, 64>
            >;

            static constexpr bool hasPerformanceHints(){
                return false;
            }
        };

        struct MultiTileConfigs{
            using type = std::tuple<
                GaplessConfig<21, Approach::kernelparamzero, Datatype::Half2, 256, 2, 4>,
                GaplessConfig<21, Approach::kernelparamzero, Datatype::Half2, 128, 2, 8>,
                GaplessConfig<21, Approach::hardcodedzero, Datatype::Half2, 128, 2, 12>,
                GaplessConfig<21, Approach::hardcodedzero, Datatype::Half2, 128, 2, 16>,
                GaplessConfig<21, Approach::hardcodedzero, Datatype::Half2, 256, 2, 20>,
                GaplessConfig<21, Approach::hardcodedzero, Datatype::Half2, 256, 2, 24>,
                GaplessConfig<21, Approach::hardcodedzero, Datatype::Half2, 256, 2, 28>,
                GaplessConfig<21, Approach::hardcodedzero, Datatype::Half2, 512, 4, 16>,
                GaplessConfig<21, Approach::hardcodedzero, Datatype::Half2, 512, 2, 36>,
                GaplessConfig<21, Approach::hardcodedzero, Datatype::Half2, 512, 2, 40>,
                GaplessConfig<21, Approach::hardcodedzero, Datatype::Half2, 512, 2, 44>,
                GaplessConfig<21, Approach::hardcodedzero, Datatype::Half2, 512, 2, 48>,
                GaplessConfig<21, Approach::kernelparamzero, Datatype::Half2, 512, 2, 52>,
                GaplessConfig<21, Approach::kernelparamzero, Datatype::Half2, 512, 4, 28>,
                GaplessConfig<21, Approach::kernelparamzero, Datatype::Half2, 512, 2, 60>,
                GaplessConfig<21, Approach::hardcodedzero, Datatype::Half2, 512, 4, 32>,
                GaplessConfig<21, Approach::hardcodedzero, Datatype::Half2, 512, 4, 36>,
                GaplessConfig<21, Approach::hardcodedzero, Datatype::Half2, 512, 4, 40>,
                GaplessConfig<21, Approach::kernelparamzero, Datatype::Half2, 512, 4, 44>,
                GaplessConfig<21, Approach::hardcodedzero, Datatype::Half2, 512, 4, 48>,
                GaplessConfig<21, Approach::kernelparamzero, Datatype::Half2, 512, 4, 52>,
                GaplessConfig<21, Approach::hardcodedzero, Datatype::Half2, 512, 8, 28>,
                GaplessConfig<21, Approach::kernelparamzero, Datatype::Half2, 512, 4, 60>,
                GaplessConfig<21, Approach::hardcodedzero, Datatype::Half2, 512, 8, 32>
            >;

            static constexpr bool hasPerformanceHints(){
                return false;
            }
        };

        return std::make_unique<MultiConfigScoreOnlyAligner<SingleTileConfigs, MultiTileConfigs>>(AlignmentAlgorithmE::Gapless);
    }

    std::unique_ptr<MultiConfigScoreOnlyAlignerInterface> make_multiconfig_gapless_aligner_amd_default(){
        return make_multiconfig_gapless_aligner_amd_gfx942();
    }

LIBMARV_NAMESPACE_END
