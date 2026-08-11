#include "score_with_endpos_alignment.cuh"
#include "score_with_endpos_alignment_interfaces.cuh"

#include <memory>

#include "../namespace.hpp"
LIBMARV_NAMESPACE_BEGIN


    std::unique_ptr<MultiConfigScoreWithEndPosAlignerInterface> make_multiconfig_sw_endpos_aligner_sm120_32bit(){
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


#if 0    
    std::unique_ptr<MultiConfigScoreWithEndPosAlignerInterface> make_multiconfig_gapless_endpos_aligner_sm120_8bit(){
        struct SingleTileConfigs{
            using type = std::tuple<
                // GaplessConfig<21,Approach::unused, Datatype::UInt8x4, 256, 4, 4>,
                // GaplessConfig<21,Approach::unused, Datatype::UInt8x4, 256, 4, 8>,
                // GaplessConfig<21,Approach::unused, Datatype::UInt8x4, 256, 4, 12>,
                // GaplessConfig<21,Approach::unused, Datatype::UInt8x4, 256, 4, 16>,
                // GaplessConfig<21,Approach::unused, Datatype::UInt8x4, 512, 4, 20>,
                // GaplessConfig<21,Approach::unused, Datatype::UInt8x4, 256, 4, 24>,
                // GaplessConfig<21,Approach::unused, Datatype::UInt8x4, 256, 4, 28>,
                // GaplessConfig<21,Approach::unused, Datatype::UInt8x4, 128, 4, 32>,
                // GaplessConfig<21,Approach::unused, Datatype::UInt8x4, 256, 4, 36>,
                // GaplessConfig<21,Approach::unused, Datatype::UInt8x4, 256, 4, 40>,
                // GaplessConfig<21,Approach::unused, Datatype::UInt8x4, 256, 4, 44>,
                // GaplessConfig<21,Approach::unused, Datatype::UInt8x4, 256, 4, 48>,
                // GaplessConfig<21,Approach::unused, Datatype::UInt8x4, 256, 4, 52>,
                // GaplessConfig<21,Approach::unused, Datatype::UInt8x4, 256, 4, 56>,
                // GaplessConfig<21,Approach::unused, Datatype::UInt8x4, 256, 4, 60>,
                GaplessConfig<21,Approach::unused, Datatype::UInt8x4, 256, 4, 64>,
                // GaplessConfig<21,Approach::unused, Datatype::UInt8x4, 256, 8, 36>,
                // GaplessConfig<21,Approach::unused, Datatype::UInt8x4, 256, 8, 40>,
                // GaplessConfig<21,Approach::unused, Datatype::UInt8x4, 256, 8, 44>,
                // GaplessConfig<21,Approach::unused, Datatype::UInt8x4, 256, 8, 48>,
                // GaplessConfig<21,Approach::unused, Datatype::UInt8x4, 256, 8, 52>,
                // GaplessConfig<21,Approach::unused, Datatype::UInt8x4, 256, 8, 56>,
                // GaplessConfig<21,Approach::unused, Datatype::UInt8x4, 256, 8, 60>,
                // GaplessConfig<21,Approach::unused, Datatype::UInt8x4, 256, 8, 64>,
                // GaplessConfig<21,Approach::unused, Datatype::UInt8x4, 256, 16, 36>,
                // GaplessConfig<21,Approach::unused, Datatype::UInt8x4, 512, 16, 40>,
                // GaplessConfig<21,Approach::unused, Datatype::UInt8x4, 512, 16, 44>,
                // GaplessConfig<21,Approach::unused, Datatype::UInt8x4, 512, 16, 48>,
                // GaplessConfig<21,Approach::unused, Datatype::UInt8x4, 512, 16, 52>,
                // GaplessConfig<21,Approach::unused, Datatype::UInt8x4, 512, 16, 56>,
                // GaplessConfig<21,Approach::unused, Datatype::UInt8x4, 512, 16, 60>,
                // GaplessConfig<21,Approach::unused, Datatype::UInt8x4, 512, 16, 64>,
                // GaplessConfig<21,Approach::unused, Datatype::UInt8x4, 512, 32, 36>,
            >;

            static constexpr bool hasPerformanceHints(){
                return false;
            }
        };
    
        struct MultiTileConfigs{
            using type = std::tuple<
               GaplessConfig<21, Approach::unused, Datatype::UInt8x4, 512, 4, 16>,
            >;

            static constexpr bool hasPerformanceHints(){
                return false;
            }
        };

        return std::make_unique<MultiConfigScoreWithEndPosAligner<SingleTileConfigs, MultiTileConfigs>>(AlignmentAlgorithmE::Gapless);
    }
#endif

    std::unique_ptr<MultiConfigScoreWithEndPosAlignerInterface> make_multiconfig_gapless_endpos_aligner_sm120_16bit(){

        struct SingleTileConfigs{
            using type = std::tuple<
                GaplessConfig<21, Approach::hardcodedzero, Datatype::Half2, 256, 4, 4>,
                GaplessConfig<21, Approach::hardcodedzero, Datatype::Half2, 256, 4, 8>,
                GaplessConfig<21, Approach::hardcodedzero, Datatype::Half2, 256, 4, 12>,
                GaplessConfig<21, Approach::hardcodedzero, Datatype::Half2, 512, 4, 16>,
                GaplessConfig<21, Approach::hardcodedzero, Datatype::Half2, 512, 4, 20>,
                GaplessConfig<21, Approach::hardcodedzero, Datatype::Half2, 256, 4, 24>,
                GaplessConfig<21, Approach::hardcodedzero, Datatype::Half2, 256, 4, 28>,
                GaplessConfig<21, Approach::hardcodedzero, Datatype::Half2, 256, 4, 32>,
                GaplessConfig<21, Approach::hardcodedzero, Datatype::Half2, 256, 4, 36>,
                GaplessConfig<21, Approach::hardcodedzero, Datatype::Half2, 256, 4, 40>,
                GaplessConfig<21, Approach::hardcodedzero, Datatype::Half2, 256, 4, 44>,
                GaplessConfig<21, Approach::hardcodedzero, Datatype::Half2, 256, 4, 48>,
                GaplessConfig<21, Approach::hardcodedzero, Datatype::Half2, 256, 4, 52>,
                GaplessConfig<21, Approach::hardcodedzero, Datatype::Half2, 256, 4, 56>,
                GaplessConfig<21, Approach::hardcodedzero, Datatype::Half2, 256, 4, 60>,
                GaplessConfig<21, Approach::hardcodedzero, Datatype::Half2, 256, 4, 64>,
                GaplessConfig<21, Approach::hardcodedzero, Datatype::Half2, 256, 8, 36>,
                GaplessConfig<21, Approach::hardcodedzero, Datatype::Half2, 256, 8, 40>,
                GaplessConfig<21, Approach::hardcodedzero, Datatype::Half2, 256, 8, 44>,
                GaplessConfig<21, Approach::hardcodedzero, Datatype::Half2, 256, 8, 48>,
                GaplessConfig<21, Approach::hardcodedzero, Datatype::Half2, 256, 8, 52>,
                GaplessConfig<21, Approach::hardcodedzero, Datatype::Half2, 256, 8, 56>,
                GaplessConfig<21, Approach::hardcodedzero, Datatype::Half2, 256, 8, 60>,
                GaplessConfig<21, Approach::hardcodedzero, Datatype::Half2, 256, 8, 64>,
                GaplessConfig<21, Approach::hardcodedzero, Datatype::Half2, 256, 16, 36>,
                GaplessConfig<21, Approach::hardcodedzero, Datatype::Half2, 512, 16, 40>,
                GaplessConfig<21, Approach::hardcodedzero, Datatype::Half2, 512, 16, 44>,
                GaplessConfig<21, Approach::hardcodedzero, Datatype::Half2, 512, 16, 48>,
                GaplessConfig<21, Approach::hardcodedzero, Datatype::Half2, 512, 16, 52>,
                GaplessConfig<21, Approach::hardcodedzero, Datatype::Half2, 512, 16, 56>,
                GaplessConfig<21, Approach::hardcodedzero, Datatype::Half2, 512, 16, 60>,
                GaplessConfig<21, Approach::hardcodedzero, Datatype::Half2, 512, 16, 64>,
                GaplessConfig<21, Approach::hardcodedzero, Datatype::Half2, 512, 32, 36>,
            >;

            static constexpr bool hasPerformanceHints(){
                return false;
            }
        };
    
        struct MultiTileConfigs{
            using type = std::tuple<
                GaplessConfig<21, Approach::hardcodedzero, Datatype::Half2, 256, 4, 32>,
                GaplessConfig<21, Approach::hardcodedzero, Datatype::Half2, 256, 4, 36>,
                GaplessConfig<21, Approach::hardcodedzero, Datatype::Half2, 256, 4, 40>,
                GaplessConfig<21, Approach::hardcodedzero, Datatype::Half2, 256, 4, 44>,
                GaplessConfig<21, Approach::hardcodedzero, Datatype::Half2, 256, 4, 48>,
                GaplessConfig<21, Approach::hardcodedzero, Datatype::Half2, 256, 4, 52>,
                GaplessConfig<21, Approach::hardcodedzero, Datatype::Half2, 256, 4, 56>,
                GaplessConfig<21, Approach::hardcodedzero, Datatype::Half2, 256, 4, 60>,
                GaplessConfig<21, Approach::hardcodedzero, Datatype::Half2, 256, 4, 64>,
                GaplessConfig<21, Approach::hardcodedzero, Datatype::Half2, 256, 8, 36>,
                GaplessConfig<21, Approach::hardcodedzero, Datatype::Half2, 256, 8, 40>,
                GaplessConfig<21, Approach::hardcodedzero, Datatype::Half2, 256, 8, 44>,
                GaplessConfig<21, Approach::hardcodedzero, Datatype::Half2, 256, 8, 48>,
                GaplessConfig<21, Approach::hardcodedzero, Datatype::Half2, 256, 8, 52>,
                GaplessConfig<21, Approach::hardcodedzero, Datatype::Half2, 256, 8, 56>,
                GaplessConfig<21, Approach::hardcodedzero, Datatype::Half2, 256, 8, 60>,
                GaplessConfig<21, Approach::hardcodedzero, Datatype::Half2, 256, 8, 64>,
                GaplessConfig<21, Approach::hardcodedzero, Datatype::Half2, 256, 16, 36>,
                GaplessConfig<21, Approach::hardcodedzero, Datatype::Half2, 512, 16, 40>,
                GaplessConfig<21, Approach::hardcodedzero, Datatype::Half2, 512, 16, 44>,
                GaplessConfig<21, Approach::hardcodedzero, Datatype::Half2, 512, 16, 48>,
                GaplessConfig<21, Approach::hardcodedzero, Datatype::Half2, 512, 16, 52>,
                GaplessConfig<21, Approach::hardcodedzero, Datatype::Half2, 512, 16, 56>,
                GaplessConfig<21, Approach::hardcodedzero, Datatype::Half2, 512, 16, 60>,
                GaplessConfig<21, Approach::hardcodedzero, Datatype::Half2, 512, 16, 64>,
                // GaplessConfig<21, Approach::hardcodedzero, Datatype::Half2, 512, 32, 36>,
            >;

            static constexpr bool hasPerformanceHints(){
                return false;
            }
        };

        return std::make_unique<MultiConfigScoreWithEndPosAligner<SingleTileConfigs, MultiTileConfigs>>(AlignmentAlgorithmE::Gapless);
    }

#if 0
    std::unique_ptr<MultiConfigScoreWithEndPosAlignerInterface> make_multiconfig_gapless_endpos_aligner_sm120_32bit(){
        struct SingleTileConfigs{
            using type = std::tuple<
                GaplessConfig<21,Approach::unused, Datatype::Float, 256, 4, 4>,
                GaplessConfig<21,Approach::unused, Datatype::Float, 256, 4, 8>,
                GaplessConfig<21,Approach::unused, Datatype::Float, 256, 4, 12>,
                GaplessConfig<21,Approach::unused, Datatype::Float, 256, 4, 16>,
                GaplessConfig<21,Approach::unused, Datatype::Float, 512, 4, 20>,
                GaplessConfig<21,Approach::unused, Datatype::Float, 256, 4, 24>,
                GaplessConfig<21,Approach::unused, Datatype::Float, 256, 4, 28>,
                GaplessConfig<21,Approach::unused, Datatype::Float, 128, 4, 32>,
                GaplessConfig<21,Approach::unused, Datatype::Float, 256, 4, 36>,
                GaplessConfig<21,Approach::unused, Datatype::Float, 256, 4, 40>,
                GaplessConfig<21,Approach::unused, Datatype::Float, 256, 4, 44>,
                GaplessConfig<21,Approach::unused, Datatype::Float, 256, 4, 48>,
                GaplessConfig<21,Approach::unused, Datatype::Float, 256, 4, 52>,
                GaplessConfig<21,Approach::unused, Datatype::Float, 256, 4, 56>,
                GaplessConfig<21,Approach::unused, Datatype::Float, 256, 4, 60>,
                GaplessConfig<21,Approach::unused, Datatype::Float, 256, 4, 64>,
                GaplessConfig<21,Approach::unused, Datatype::Float, 256, 8, 36>,
                GaplessConfig<21,Approach::unused, Datatype::Float, 256, 8, 40>,
                GaplessConfig<21,Approach::unused, Datatype::Float, 256, 8, 44>,
                GaplessConfig<21,Approach::unused, Datatype::Float, 256, 8, 48>,
                GaplessConfig<21,Approach::unused, Datatype::Float, 256, 8, 52>,
                GaplessConfig<21,Approach::unused, Datatype::Float, 256, 8, 56>,
                GaplessConfig<21,Approach::unused, Datatype::Float, 256, 8, 60>,
                GaplessConfig<21,Approach::unused, Datatype::Float, 256, 8, 64>,
                GaplessConfig<21,Approach::unused, Datatype::Float, 256, 16, 36>,
                GaplessConfig<21,Approach::unused, Datatype::Float, 512, 16, 40>,
                GaplessConfig<21,Approach::unused, Datatype::Float, 512, 16, 44>,
                GaplessConfig<21,Approach::unused, Datatype::Float, 512, 16, 48>,
                GaplessConfig<21,Approach::unused, Datatype::Float, 512, 16, 52>,
                GaplessConfig<21,Approach::unused, Datatype::Float, 512, 16, 56>,
                GaplessConfig<21,Approach::unused, Datatype::Float, 512, 16, 60>,
                GaplessConfig<21,Approach::unused, Datatype::Float, 512, 16, 64>,
                GaplessConfig<21,Approach::unused, Datatype::Float, 512, 32, 36>,
            >;

            static constexpr bool hasPerformanceHints(){
                return false;
            }
        };
    
        struct MultiTileConfigs{
            using type = std::tuple<
               GaplessConfig<21, Approach::unused, Datatype::Float, 512, 4, 64>,
            >;

            static constexpr bool hasPerformanceHints(){
                return false;
            }
        };   
                    
        return std::make_unique<MultiConfigScoreWithEndPosAligner<SingleTileConfigs, MultiTileConfigs>>(AlignmentAlgorithmE::Gapless);
    }
#endif



LIBMARV_NAMESPACE_END