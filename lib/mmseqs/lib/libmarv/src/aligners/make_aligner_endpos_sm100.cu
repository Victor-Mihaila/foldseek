#include "score_with_endpos_alignment.cuh"
#include "score_with_endpos_alignment_interfaces.cuh"

#include <memory>

#include "../namespace.hpp"
LIBMARV_NAMESPACE_BEGIN


    std::unique_ptr<MultiConfigScoreWithEndPosAlignerInterface> make_multiconfig_sw_endpos_aligner_sm100_32bit(){
        //NOTE: copied from a different arch,
        struct SingleTileConfigs{
            using type = std::tuple<
                SWConfig<21, Approach::unused, Datatype::Int, 512, 4, 4>,
                SWConfig<21, Approach::unused, Datatype::Int, 512, 4, 8>,
                SWConfig<21, Approach::unused, Datatype::Int, 512, 4, 12>,
                SWConfig<21, Approach::unused, Datatype::Int, 512, 4, 16>,
                SWConfig<21, Approach::unused, Datatype::Int, 512, 4, 20>,
                SWConfig<21, Approach::unused, Datatype::Int, 512, 4, 24>,
                SWConfig<21, Approach::unused, Datatype::Int, 512, 4, 28>,
                SWConfig<21, Approach::unused, Datatype::Int, 512, 4, 32>,
                SWConfig<21, Approach::unused, Datatype::Int, 512, 4, 36>,
                SWConfig<21, Approach::unused, Datatype::Int, 512, 8, 20>,
                SWConfig<21, Approach::unused, Datatype::Int, 512, 4, 44>,
                SWConfig<21, Approach::unused, Datatype::Int, 512, 8, 24>,
                SWConfig<21, Approach::unused, Datatype::Int, 512, 8, 28>,
                SWConfig<21, Approach::unused, Datatype::Int, 512, 8, 32>,
                SWConfig<21, Approach::unused, Datatype::Int, 512, 8, 36>,
                SWConfig<21, Approach::unused, Datatype::Int, 512, 16, 20>,
                SWConfig<21, Approach::unused, Datatype::Int, 512, 8, 44>,
                SWConfig<21, Approach::unused, Datatype::Int, 512, 16, 24>,
                SWConfig<21, Approach::unused, Datatype::Int, 512, 16, 28>,
                SWConfig<21, Approach::unused, Datatype::Int, 512, 16, 32>,
                SWConfig<21, Approach::unused, Datatype::Int, 512, 16, 36>,
                SWConfig<21, Approach::unused, Datatype::Int, 512, 32, 20>,
                SWConfig<21, Approach::unused, Datatype::Int, 512, 16, 44>,
                SWConfig<21, Approach::unused, Datatype::Int, 512, 32, 24>,
                SWConfig<21, Approach::unused, Datatype::Int, 512, 32, 28>,
                SWConfig<21, Approach::unused, Datatype::Int, 512, 32, 32>
            >;

            static constexpr bool hasPerformanceHints(){
                return false;
            }
        };
    
        struct MultiTileConfigs{
            using type = std::tuple<
                SWConfig<21, Approach::unused, Datatype::Int, 512, 8, 32>,
                SWConfig<21, Approach::unused, Datatype::Int, 512, 8, 36>,
                SWConfig<21, Approach::unused, Datatype::Int, 512, 16, 20>,
                SWConfig<21, Approach::unused, Datatype::Int, 512, 8, 44>,
                SWConfig<21, Approach::unused, Datatype::Int, 512, 16, 24>,
                SWConfig<21, Approach::unused, Datatype::Int, 512, 16, 28>,
                SWConfig<21, Approach::unused, Datatype::Int, 512, 16, 32>,
                SWConfig<21, Approach::unused, Datatype::Int, 512, 16, 36>,
                SWConfig<21, Approach::unused, Datatype::Int, 512, 32, 20>,
                SWConfig<21, Approach::unused, Datatype::Int, 512, 16, 44>,
                SWConfig<21, Approach::unused, Datatype::Int, 512, 32, 24>,
                SWConfig<21, Approach::unused, Datatype::Int, 512, 32, 28>,
                SWConfig<21, Approach::unused, Datatype::Int, 512, 32, 32>
            >;

            static constexpr bool hasPerformanceHints(){
                return false;
            }
        };
    
                    
        return std::make_unique<MultiConfigScoreWithEndPosAligner<SingleTileConfigs, MultiTileConfigs>>(AlignmentAlgorithmE::SW);
    }

LIBMARV_NAMESPACE_END