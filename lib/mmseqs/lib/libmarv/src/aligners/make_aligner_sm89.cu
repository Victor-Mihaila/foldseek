#include "score_only_alignment.cuh"
#include "score_only_alignment_interfaces.cuh"
#include "score_with_endpos_alignment.cuh"
#include "score_with_endpos_alignment_interfaces.cuh"

#include <memory>

#include "../namespace.hpp"
LIBMARV_NAMESPACE_BEGIN


    std::unique_ptr<MultiConfigScoreOnlyAlignerInterface> make_multiconfig_gapless_aligner_sm89_16bit(){
        struct SingleTileConfigs{
            using type = std::tuple<
                GaplessConfig<21, Approach::kernelparamzero, Datatype::Half2, 256, 4, 4>,
                GaplessConfig<21, Approach::kernelparamzero, Datatype::Half2, 256, 4, 8>,
                GaplessConfig<21, Approach::hardcodedzero, Datatype::Half2, 256, 4, 12>,
                GaplessConfig<21, Approach::kernelparamzero, Datatype::Half2, 512, 4, 16>,
                GaplessConfig<21, Approach::hardcodedzero, Datatype::Half2, 128, 4, 20>,
                GaplessConfig<21, Approach::kernelparamzero, Datatype::Half2, 128, 4, 24>,
                GaplessConfig<21, Approach::hardcodedzero, Datatype::Half2, 512, 4, 28>,
                GaplessConfig<21, Approach::kernelparamzero, Datatype::Half2, 512, 4, 32>,
                GaplessConfig<21, Approach::hardcodedzero, Datatype::Half2, 512, 4, 36>,
                GaplessConfig<21, Approach::kernelparamzero, Datatype::Half2, 128, 4, 40>,
                GaplessConfig<21, Approach::kernelparamzero, Datatype::Half2, 128, 4, 44>,
                GaplessConfig<21, Approach::hardcodedzero, Datatype::Half2, 256, 4, 48>,
                GaplessConfig<21, Approach::hardcodedzero, Datatype::Half2, 128, 4, 52>,
                GaplessConfig<21, Approach::kernelparamzero, Datatype::Half2, 128, 8, 28>,
                GaplessConfig<21, Approach::kernelparamzero, Datatype::Half2, 256, 4, 60>,
                GaplessConfig<21, Approach::hardcodedzero, Datatype::Half2, 256, 4, 64>,
                GaplessConfig<21, Approach::kernelparamzero, Datatype::Half2, 128, 8, 36>,
                GaplessConfig<21, Approach::hardcodedzero, Datatype::Half2, 128, 8, 40>,
                GaplessConfig<21, Approach::hardcodedzero, Datatype::Half2, 128, 8, 44>,
                GaplessConfig<21, Approach::kernelparamzero, Datatype::Half2, 128, 8, 48>,
                GaplessConfig<21, Approach::hardcodedzero, Datatype::Half2, 128, 8, 52>,
                GaplessConfig<21, Approach::hardcodedzero, Datatype::Half2, 256, 8, 56>,
                GaplessConfig<21, Approach::kernelparamzero, Datatype::Half2, 256, 8, 60>,
                GaplessConfig<21, Approach::hardcodedzero, Datatype::Half2, 512, 8, 64>,
                GaplessConfig<21, Approach::kernelparamzero, Datatype::Half2, 512, 16, 36>,
                GaplessConfig<21, Approach::hardcodedzero, Datatype::Half2, 256, 16, 40>,
                GaplessConfig<21, Approach::hardcodedzero, Datatype::Half2, 256, 16, 44>,
                GaplessConfig<21, Approach::hardcodedzero, Datatype::Half2, 512, 16, 48>,
                GaplessConfig<21, Approach::hardcodedzero, Datatype::Half2, 512, 16, 52>,
                GaplessConfig<21, Approach::kernelparamzero, Datatype::Half2, 512, 16, 56>,
                GaplessConfig<21, Approach::hardcodedzero, Datatype::Half2, 512, 16, 60>,
                GaplessConfig<21, Approach::hardcodedzero, Datatype::Half2, 512, 16, 64>,
                GaplessConfig<21, Approach::kernelparamzero, Datatype::Half2, 256, 32, 36>,
            >;


            static constexpr auto getPerformanceHints(){
                constexpr auto result = std::array{7564,11363,13241,13535,13099,14057,15027,15300,
                    15820,16334,16295,16412,15917,15225,17017,16567,16100,16465,16316,16738,15908,
                    15710,16785,17053,15612,15962,15497,15330,16448,16376,16669,16955,15772,};

                static_assert(result.size() == std::tuple_size_v<type>);

                return result;
            };

            static constexpr bool hasPerformanceHints(){
                return true;
            }
        };
    
        struct MultiTileConfigs{
            using type = std::tuple<
                GaplessConfig<21, Approach::hardcodedzero, Datatype::Half2, 512, 4, 32>,
                GaplessConfig<21, Approach::hardcodedzero, Datatype::Half2, 512, 4, 36>,
                GaplessConfig<21, Approach::kernelparamzero, Datatype::Half2, 512, 4, 40>,
                GaplessConfig<21, Approach::hardcodedzero, Datatype::Half2, 512, 4, 44>,
                GaplessConfig<21, Approach::hardcodedzero, Datatype::Half2, 512, 4, 48>,
                GaplessConfig<21, Approach::hardcodedzero, Datatype::Half2, 512, 4, 52>,
                GaplessConfig<21, Approach::hardcodedzero, Datatype::Half2, 512, 4, 56>,
                GaplessConfig<21, Approach::hardcodedzero, Datatype::Half2, 512, 4, 60>,
                GaplessConfig<21, Approach::hardcodedzero, Datatype::Half2, 512, 4, 64>,
                GaplessConfig<21, Approach::hardcodedzero, Datatype::Half2, 512, 8, 36>,
                GaplessConfig<21, Approach::hardcodedzero, Datatype::Half2, 512, 8, 40>,
                GaplessConfig<21, Approach::kernelparamzero, Datatype::Half2, 256, 8, 44>,
                GaplessConfig<21, Approach::kernelparamzero, Datatype::Half2, 512, 8, 48>,
                GaplessConfig<21, Approach::kernelparamzero, Datatype::Half2, 512, 8, 52>,
                GaplessConfig<21, Approach::hardcodedzero, Datatype::Half2, 512, 8, 56>,
                GaplessConfig<21, Approach::kernelparamzero, Datatype::Half2, 512, 8, 60>,
                GaplessConfig<21, Approach::kernelparamzero, Datatype::Half2, 512, 8, 64>,
                GaplessConfig<21, Approach::kernelparamzero, Datatype::Half2, 512, 16, 36>,
                GaplessConfig<21, Approach::kernelparamzero, Datatype::Half2, 512, 16, 40>,
                GaplessConfig<21, Approach::kernelparamzero, Datatype::Half2, 512, 16, 44>,
                GaplessConfig<21, Approach::kernelparamzero, Datatype::Half2, 512, 16, 48>,
                GaplessConfig<21, Approach::hardcodedzero, Datatype::Half2, 512, 16, 52>,
                GaplessConfig<21, Approach::kernelparamzero, Datatype::Half2, 512, 16, 56>,
                GaplessConfig<21, Approach::kernelparamzero, Datatype::Half2, 512, 16, 60>,
                GaplessConfig<21, Approach::hardcodedzero, Datatype::Half2, 512, 16, 64>,
                GaplessConfig<21, Approach::hardcodedzero, Datatype::Half2, 512, 32, 36>,
            >;


            static constexpr auto getPerformanceHints(){
                constexpr auto result = std::array{13749,14007,14380,14450,14681,15058,15305,
                    14927,15446,14977,15091,15467,15574,15888,15741,16056,16267,15495,
                    15816,16020,15886,16241,16235,16360,16622,16087,};

                static_assert(result.size() == std::tuple_size_v<type>);

                return result;
            };

            static constexpr bool hasPerformanceHints(){
                return true;
            }
        };
    
                    
        return std::make_unique<MultiConfigScoreOnlyAligner<SingleTileConfigs, MultiTileConfigs>>(AlignmentAlgorithmE::Gapless);
    }


LIBMARV_NAMESPACE_END