#include "score_only_alignment.cuh"
#include "score_only_alignment_interfaces.cuh"
#include "score_with_endpos_alignment.cuh"
#include "score_with_endpos_alignment_interfaces.cuh"

#include <memory>

#include "../namespace.hpp"
LIBMARV_NAMESPACE_BEGIN

#if defined(__CUDACC__) && !(__CUDACC_VER_MAJOR__ < 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ < 2))
    std::unique_ptr<MultiConfigScoreOnlyAlignerInterface> make_multiconfig_gapless_aligner_sm120_8bit(){

        struct SingleTileConfigs{
            using type = std::tuple<
                GaplessConfig<21,Approach::unused, Datatype::UInt8x4, 256, 4, 4>,
                GaplessConfig<21,Approach::unused, Datatype::UInt8x4, 256, 4, 8>,
                GaplessConfig<21,Approach::unused, Datatype::UInt8x4, 256, 4, 12>,
                GaplessConfig<21,Approach::unused, Datatype::UInt8x4, 256, 4, 16>,
                GaplessConfig<21,Approach::unused, Datatype::UInt8x4, 512, 4, 20>,
                GaplessConfig<21,Approach::unused, Datatype::UInt8x4, 256, 4, 24>,
                GaplessConfig<21,Approach::unused, Datatype::UInt8x4, 256, 4, 28>,
                GaplessConfig<21,Approach::unused, Datatype::UInt8x4, 128, 4, 32>,
                GaplessConfig<21,Approach::unused, Datatype::UInt8x4, 256, 4, 36>,
                GaplessConfig<21,Approach::unused, Datatype::UInt8x4, 256, 4, 40>,
                GaplessConfig<21,Approach::unused, Datatype::UInt8x4, 256, 4, 44>,
                GaplessConfig<21,Approach::unused, Datatype::UInt8x4, 256, 4, 48>,
                GaplessConfig<21,Approach::unused, Datatype::UInt8x4, 256, 4, 52>,
                GaplessConfig<21,Approach::unused, Datatype::UInt8x4, 256, 4, 56>,
                GaplessConfig<21,Approach::unused, Datatype::UInt8x4, 256, 4, 60>,
                GaplessConfig<21,Approach::unused, Datatype::UInt8x4, 256, 4, 64>,
                GaplessConfig<21,Approach::unused, Datatype::UInt8x4, 256, 8, 36>,
                GaplessConfig<21,Approach::unused, Datatype::UInt8x4, 256, 8, 40>,
                GaplessConfig<21,Approach::unused, Datatype::UInt8x4, 256, 8, 44>,
                GaplessConfig<21,Approach::unused, Datatype::UInt8x4, 256, 8, 48>,
                GaplessConfig<21,Approach::unused, Datatype::UInt8x4, 256, 8, 52>,
                GaplessConfig<21,Approach::unused, Datatype::UInt8x4, 256, 8, 56>,
                GaplessConfig<21,Approach::unused, Datatype::UInt8x4, 256, 8, 60>,
                GaplessConfig<21,Approach::unused, Datatype::UInt8x4, 256, 8, 64>,
                GaplessConfig<21,Approach::unused, Datatype::UInt8x4, 256, 16, 36>,
                GaplessConfig<21,Approach::unused, Datatype::UInt8x4, 512, 16, 40>,
                GaplessConfig<21,Approach::unused, Datatype::UInt8x4, 512, 16, 44>,
                GaplessConfig<21,Approach::unused, Datatype::UInt8x4, 512, 16, 48>,
                GaplessConfig<21,Approach::unused, Datatype::UInt8x4, 512, 16, 52>,
                GaplessConfig<21,Approach::unused, Datatype::UInt8x4, 512, 16, 56>,
                GaplessConfig<21,Approach::unused, Datatype::UInt8x4, 512, 16, 60>,
                GaplessConfig<21,Approach::unused, Datatype::UInt8x4, 512, 16, 64>,
                GaplessConfig<21,Approach::unused, Datatype::UInt8x4, 512, 32, 36>,
            >;

            static constexpr auto getPerformanceHints(){
                constexpr auto result = std::array{26258,33804,38168,40405,42943,44209,45167,45178,47466,
                    47379,47432,48142,48963,49607,49607,50046,48108,48327,47959,48725,49383,50148,49808,50444,
                    47593,47702,47773,47697,48978,50092,48573,50152,47702};

                static_assert(result.size() == std::tuple_size_v<type>);

                return result;
            };

            static constexpr bool hasPerformanceHints(){
                return true;
            }

            
        };
        
    
        struct MultiTileConfigs{
            using type = std::tuple<
                GaplessConfig<21, Approach::unused, Datatype::UInt8x4, 128, 4, 32>, 
                GaplessConfig<21, Approach::unused, Datatype::UInt8x4, 128, 4, 36>, 
                GaplessConfig<21, Approach::unused, Datatype::UInt8x4, 256, 4, 40>, 
                GaplessConfig<21, Approach::unused, Datatype::UInt8x4, 512, 4, 44>, 
                GaplessConfig<21, Approach::unused, Datatype::UInt8x4, 128, 4, 48>, 
                GaplessConfig<21, Approach::unused, Datatype::UInt8x4, 512, 4, 52>, 
                GaplessConfig<21, Approach::unused, Datatype::UInt8x4, 512, 4, 56>, 
                GaplessConfig<21, Approach::unused, Datatype::UInt8x4, 512, 4, 60>, 
                GaplessConfig<21, Approach::unused, Datatype::UInt8x4, 128, 8, 32>, 
                GaplessConfig<21, Approach::unused, Datatype::UInt8x4, 128, 8, 36>, 
                GaplessConfig<21, Approach::unused, Datatype::UInt8x4, 128, 8, 40>, 
                GaplessConfig<21, Approach::unused, Datatype::UInt8x4, 128, 8, 44>, 
                GaplessConfig<21, Approach::unused, Datatype::UInt8x4, 128, 8, 48>, 
                GaplessConfig<21, Approach::unused, Datatype::UInt8x4, 512, 8, 52>, 
                GaplessConfig<21, Approach::unused, Datatype::UInt8x4, 128, 8, 56>, 
                GaplessConfig<21, Approach::unused, Datatype::UInt8x4, 128, 8, 60>, 
                GaplessConfig<21, Approach::unused, Datatype::UInt8x4, 512, 8, 64>, 
                GaplessConfig<21, Approach::unused, Datatype::UInt8x4, 256, 16, 36>, 
                GaplessConfig<21, Approach::unused, Datatype::UInt8x4, 512, 16, 40>, 
                GaplessConfig<21, Approach::unused, Datatype::UInt8x4, 512, 16, 44>, 
                GaplessConfig<21, Approach::unused, Datatype::UInt8x4, 512, 16, 48>, 
                GaplessConfig<21, Approach::unused, Datatype::UInt8x4, 512, 16, 52>, 
                GaplessConfig<21, Approach::unused, Datatype::UInt8x4, 512, 16, 56>, 
                GaplessConfig<21, Approach::unused, Datatype::UInt8x4, 256, 16, 60>, 
                GaplessConfig<21, Approach::unused, Datatype::UInt8x4, 512, 16, 64>, 
                GaplessConfig<21, Approach::unused, Datatype::UInt8x4, 512, 32, 36>, 
            >;

            static constexpr auto getPerformanceHints(){
                constexpr auto result = std::array{43485,44907,
                    43469,44302,44207,44861,44024,42347,44803,45920,45841,46194,46474,44034,44043,
                    43945,41950,41922,42657,42826,43669,43716,43714,39791,42239,42729};

                static_assert(result.size() == std::tuple_size_v<type>);

                return result;
            };

            static constexpr bool hasPerformanceHints(){
                return true;
            }

            
        };
    
                    
        return std::make_unique<MultiConfigScoreOnlyAligner<SingleTileConfigs, MultiTileConfigs>>(AlignmentAlgorithmE::Gapless);
    }
#endif    

    std::unique_ptr<MultiConfigScoreOnlyAlignerInterface> make_multiconfig_gapless_aligner_sm120_16bit(){
        struct SingleTileConfigs{
            using type = std::tuple<
                GaplessConfig<21, Approach::hardcodedzero, Datatype::Short2, 256, 4, 4>,
                GaplessConfig<21, Approach::hardcodedzero, Datatype::Short2, 256, 4, 8>,
                GaplessConfig<21, Approach::hardcodedzero, Datatype::Short2, 256, 4, 12>,
                GaplessConfig<21, Approach::hardcodedzero, Datatype::Short2, 512, 4, 16>,
                GaplessConfig<21, Approach::hardcodedzero, Datatype::Short2, 512, 4, 20>,
                GaplessConfig<21, Approach::kernelparamzero, Datatype::Short2, 256, 4, 24>,
                GaplessConfig<21, Approach::kernelparamzero, Datatype::Short2, 256, 4, 28>,
                GaplessConfig<21, Approach::hardcodedzero, Datatype::Short2, 256, 4, 32>,
                GaplessConfig<21, Approach::hardcodedzero, Datatype::Short2, 256, 4, 36>,
                GaplessConfig<21, Approach::hardcodedzero, Datatype::Short2, 256, 4, 40>,
                GaplessConfig<21, Approach::kernelparamzero, Datatype::Short2, 256, 4, 44>,
                GaplessConfig<21, Approach::hardcodedzero, Datatype::Short2, 256, 4, 48>,
                GaplessConfig<21, Approach::hardcodedzero, Datatype::Short2, 256, 4, 52>,
                GaplessConfig<21, Approach::kernelparamzero, Datatype::Short2, 256, 4, 56>,
                GaplessConfig<21, Approach::hardcodedzero, Datatype::Short2, 256, 4, 60>,
                GaplessConfig<21, Approach::hardcodedzero, Datatype::Short2, 256, 4, 64>,
                GaplessConfig<21, Approach::hardcodedzero, Datatype::Short2, 256, 8, 36>,
                GaplessConfig<21, Approach::hardcodedzero, Datatype::Short2, 256, 8, 40>,
                GaplessConfig<21, Approach::kernelparamzero, Datatype::Short2, 256, 8, 44>,
                GaplessConfig<21, Approach::hardcodedzero, Datatype::Short2, 256, 8, 48>,
                GaplessConfig<21, Approach::hardcodedzero, Datatype::Short2, 256, 8, 52>,
                GaplessConfig<21, Approach::hardcodedzero, Datatype::Short2, 256, 8, 56>,
                GaplessConfig<21, Approach::hardcodedzero, Datatype::Short2, 256, 8, 60>,
                GaplessConfig<21, Approach::hardcodedzero, Datatype::Short2, 256, 8, 64>,
                GaplessConfig<21, Approach::hardcodedzero, Datatype::Short2, 256, 16, 36>,
                GaplessConfig<21, Approach::hardcodedzero, Datatype::Short2, 512, 16, 40>,
                GaplessConfig<21, Approach::kernelparamzero, Datatype::Short2, 512, 16, 44>,
                GaplessConfig<21, Approach::hardcodedzero, Datatype::Short2, 512, 16, 48>,
                GaplessConfig<21, Approach::hardcodedzero, Datatype::Short2, 512, 16, 52>,
                GaplessConfig<21, Approach::hardcodedzero, Datatype::Short2, 512, 16, 56>,
                GaplessConfig<21, Approach::kernelparamzero, Datatype::Short2, 512, 16, 60>,
                GaplessConfig<21, Approach::hardcodedzero, Datatype::Short2, 512, 16, 64>,
                GaplessConfig<21, Approach::hardcodedzero, Datatype::Short2, 512, 32, 36>,
            >;

            static constexpr auto getPerformanceHints(){
                constexpr auto result = std::array{12736,17343,19042,20001,21370,21701,22414,
                    22963,22821,23135,23350,23584,23802,24125,24147,24181,23352,23698,23745,
                    24005,24118,24362,24425,24383,23502,23639,23649,23757,24038,24352,24201,24343,23743,};

                static_assert(result.size() == std::tuple_size_v<type>);

                return result;
            };

            static constexpr bool hasPerformanceHints(){
                return false;
            }
        };
    
        struct MultiTileConfigs{
            using type = std::tuple<
                GaplessConfig<21, Approach::hardcodedzero, Datatype::Short2, 512, 4, 32>,
                GaplessConfig<21, Approach::hardcodedzero, Datatype::Short2, 512, 4, 36>,
                GaplessConfig<21, Approach::kernelparamzero, Datatype::Short2, 512, 4, 40>,
                GaplessConfig<21, Approach::kernelparamzero, Datatype::Short2, 512, 4, 44>,
                GaplessConfig<21, Approach::hardcodedzero, Datatype::Short2, 512, 4, 48>,
                GaplessConfig<21, Approach::kernelparamzero, Datatype::Short2, 512, 4, 52>,
                GaplessConfig<21, Approach::hardcodedzero, Datatype::Short2, 512, 4, 56>,
                GaplessConfig<21, Approach::hardcodedzero, Datatype::Short2, 512, 4, 60>,
                GaplessConfig<21, Approach::hardcodedzero, Datatype::Short2, 512, 4, 64>,
                GaplessConfig<21, Approach::hardcodedzero, Datatype::Short2, 256, 8, 36>,
                GaplessConfig<21, Approach::hardcodedzero, Datatype::Short2, 256, 8, 40>,
                GaplessConfig<21, Approach::hardcodedzero, Datatype::Short2, 256, 8, 44>,
                GaplessConfig<21, Approach::hardcodedzero, Datatype::Short2, 256, 8, 48>,
                GaplessConfig<21, Approach::hardcodedzero, Datatype::Short2, 256, 8, 52>,
                GaplessConfig<21, Approach::hardcodedzero, Datatype::Short2, 512, 8, 56>,
                GaplessConfig<21, Approach::hardcodedzero, Datatype::Short2, 256, 8, 60>,
                GaplessConfig<21, Approach::hardcodedzero, Datatype::Short2, 256, 8, 64>,
                GaplessConfig<21, Approach::hardcodedzero, Datatype::Short2, 512, 16, 36>,
                GaplessConfig<21, Approach::hardcodedzero, Datatype::Short2, 512, 16, 40>,
                GaplessConfig<21, Approach::hardcodedzero, Datatype::Short2, 512, 16, 44>,
                GaplessConfig<21, Approach::hardcodedzero, Datatype::Short2, 512, 16, 48>,
                GaplessConfig<21, Approach::hardcodedzero, Datatype::Short2, 512, 16, 52>,
                GaplessConfig<21, Approach::hardcodedzero, Datatype::Short2, 512, 16, 56>,
                GaplessConfig<21, Approach::hardcodedzero, Datatype::Short2, 512, 16, 60>,
                GaplessConfig<21, Approach::hardcodedzero, Datatype::Short2, 512, 16, 64>,
                GaplessConfig<21, Approach::hardcodedzero, Datatype::Half2, 512, 32, 36>,
            >;


            static constexpr auto getPerformanceHints(){
                constexpr auto result = std::array{21486,22225,22598,22864,23136,23653,24032,
                    23747,23134,22751,23014,23341,23393,23601,23659,23346,23272,22331,22731,
                    22716,22834,22980,23256,22624,22714,22338,};

                static_assert(result.size() == std::tuple_size_v<type>);

                return result;
            };

            static constexpr bool hasPerformanceHints(){
                return false;
            }
        };
    
                    
        return std::make_unique<MultiConfigScoreOnlyAligner<SingleTileConfigs, MultiTileConfigs>>(AlignmentAlgorithmE::Gapless);
    }

#if 0
    std::unique_ptr<MultiConfigScoreOnlyAlignerInterface> make_multiconfig_gapless_aligner_sm120_32bit(){
        struct SingleTileConfigs{
            using type = std::tuple<
            GaplessConfig<21, Approach::unused, Datatype::Float, 512, 2, 4>,
            GaplessConfig<21, Approach::unused, Datatype::Float, 512, 2, 6>,
            GaplessConfig<21, Approach::unused, Datatype::Float, 512, 2, 8>,
            GaplessConfig<21, Approach::unused, Datatype::Float, 512, 2, 10>,
            GaplessConfig<21, Approach::unused, Datatype::Float, 512, 2, 12>,
            GaplessConfig<21, Approach::unused, Datatype::Float, 512, 2, 14>,
            GaplessConfig<21, Approach::unused, Datatype::Float, 512, 2, 16>,
            GaplessConfig<21, Approach::unused, Datatype::Float, 512, 2, 18>,
            GaplessConfig<21, Approach::unused, Datatype::Float, 256, 2, 20>,
            GaplessConfig<21, Approach::unused, Datatype::Float, 256, 2, 22>,
            GaplessConfig<21, Approach::unused, Datatype::Float, 256, 2, 24>,
            GaplessConfig<21, Approach::unused, Datatype::Float, 256, 2, 26>,
            GaplessConfig<21, Approach::unused, Datatype::Float, 256, 2, 28>,
            GaplessConfig<21, Approach::unused, Datatype::Float, 256, 2, 30>,
            GaplessConfig<21, Approach::unused, Datatype::Float, 256, 2, 32>,
            GaplessConfig<21, Approach::unused, Datatype::Float, 256, 2, 34>,
            GaplessConfig<21, Approach::unused, Datatype::Float, 256, 2, 36>,
            GaplessConfig<21, Approach::unused, Datatype::Float, 512, 2, 38>,
            GaplessConfig<21, Approach::unused, Datatype::Float, 512, 2, 40>,
            GaplessConfig<21, Approach::unused, Datatype::Float, 512, 2, 42>,
            GaplessConfig<21, Approach::unused, Datatype::Float, 512, 2, 44>,
            GaplessConfig<21, Approach::unused, Datatype::Float, 128, 2, 46>,
            GaplessConfig<21, Approach::unused, Datatype::Float, 512, 2, 48>,
            GaplessConfig<21, Approach::unused, Datatype::Float, 512, 2, 50>,
            GaplessConfig<21, Approach::unused, Datatype::Float, 512, 2, 52>,
            GaplessConfig<21, Approach::unused, Datatype::Float, 512, 2, 54>,
            
            GaplessConfig<21, Approach::unused, Datatype::Float, 128, 4, 28>,
            GaplessConfig<21, Approach::unused, Datatype::Float, 512, 2, 58>,
            GaplessConfig<21, Approach::unused, Datatype::Float, 128, 4, 30>,
            GaplessConfig<21, Approach::unused, Datatype::Float, 512, 2, 62>,
            GaplessConfig<21, Approach::unused, Datatype::Float, 128, 4, 32>,
            GaplessConfig<21, Approach::unused, Datatype::Float, 128, 4, 34>,
            GaplessConfig<21, Approach::unused, Datatype::Float, 128, 4, 36>,
            GaplessConfig<21, Approach::unused, Datatype::Float, 128, 4, 38>,
            GaplessConfig<21, Approach::unused, Datatype::Float, 128, 4, 40>,
            GaplessConfig<21, Approach::unused, Datatype::Float, 128, 4, 42>,
            GaplessConfig<21, Approach::unused, Datatype::Float, 128, 4, 44>,
            GaplessConfig<21, Approach::unused, Datatype::Float, 128, 4, 46>,
            GaplessConfig<21, Approach::unused, Datatype::Float, 512, 8, 24>,
            GaplessConfig<21, Approach::unused, Datatype::Float, 512, 4, 50>,
            GaplessConfig<21, Approach::unused, Datatype::Float, 512, 8, 26>,
            GaplessConfig<21, Approach::unused, Datatype::Float, 512, 4, 54>,
            GaplessConfig<21, Approach::unused, Datatype::Float, 512, 4, 56>,
            GaplessConfig<21, Approach::unused, Datatype::Float, 512, 4, 58>,
            GaplessConfig<21, Approach::unused, Datatype::Float, 512, 4, 60>,
            GaplessConfig<21, Approach::unused, Datatype::Float, 512, 4, 62>,
            GaplessConfig<21, Approach::unused, Datatype::Float, 256, 4, 64>,
            GaplessConfig<21, Approach::unused, Datatype::Float, 128, 8, 34>,
            GaplessConfig<21, Approach::unused, Datatype::Float, 256, 8, 36>,
            GaplessConfig<21, Approach::unused, Datatype::Float, 256, 8, 38>,
            GaplessConfig<21, Approach::unused, Datatype::Float, 128, 8, 40>,
            GaplessConfig<21, Approach::unused, Datatype::Float, 128, 8, 42>,
            GaplessConfig<21, Approach::unused, Datatype::Float, 128, 8, 44>,
            GaplessConfig<21, Approach::unused, Datatype::Float, 128, 8, 46>,
            GaplessConfig<21, Approach::unused, Datatype::Float, 256, 16, 24>,
            GaplessConfig<21, Approach::unused, Datatype::Float, 512, 8, 50>,
            GaplessConfig<21, Approach::unused, Datatype::Float, 512, 8, 52>,
            GaplessConfig<21, Approach::unused, Datatype::Float, 512, 8, 54>,
            GaplessConfig<21, Approach::unused, Datatype::Float, 512, 8, 56>,
            GaplessConfig<21, Approach::unused, Datatype::Float, 512, 8, 58>,
            GaplessConfig<21, Approach::unused, Datatype::Float, 512, 8, 60>,
            GaplessConfig<21, Approach::unused, Datatype::Float, 512, 8, 62>,
            GaplessConfig<21, Approach::unused, Datatype::Float, 512, 8, 64>,
            GaplessConfig<21, Approach::unused, Datatype::Float, 256, 16, 34>,
            GaplessConfig<21, Approach::unused, Datatype::Float, 256, 16, 36>,
            GaplessConfig<21, Approach::unused, Datatype::Float, 256, 16, 38>,
            GaplessConfig<21, Approach::unused, Datatype::Float, 512, 32, 20>,
            GaplessConfig<21, Approach::unused, Datatype::Float, 512, 16, 42>,
            GaplessConfig<21, Approach::unused, Datatype::Float, 512, 32, 22>,
            GaplessConfig<21, Approach::unused, Datatype::Float, 512, 16, 46>,
            GaplessConfig<21, Approach::unused, Datatype::Float, 512, 32, 24>,
            GaplessConfig<21, Approach::unused, Datatype::Float, 512, 16, 50>,
            GaplessConfig<21, Approach::unused, Datatype::Float, 512, 32, 26>,
            GaplessConfig<21, Approach::unused, Datatype::Float, 256, 16, 54>,
            GaplessConfig<21, Approach::unused, Datatype::Float, 512, 16, 56>,
            GaplessConfig<21, Approach::unused, Datatype::Float, 256, 16, 58>,
            GaplessConfig<21, Approach::unused, Datatype::Float, 256, 16, 60>,
            GaplessConfig<21, Approach::unused, Datatype::Float, 512, 16, 62>,
            GaplessConfig<21, Approach::unused, Datatype::Float, 512, 16, 64>,
            GaplessConfig<21, Approach::unused, Datatype::Float, 512, 32, 34>,
            GaplessConfig<21, Approach::unused, Datatype::Float, 512, 32, 36>,
            GaplessConfig<21, Approach::unused, Datatype::Float, 512, 32, 38>,
            GaplessConfig<21, Approach::unused, Datatype::Float, 512, 32, 40>,
            GaplessConfig<21, Approach::unused, Datatype::Float, 512, 32, 42>,
            GaplessConfig<21, Approach::unused, Datatype::Float, 512, 32, 44>,
            GaplessConfig<21, Approach::unused, Datatype::Float, 512, 32, 46>,
            GaplessConfig<21, Approach::unused, Datatype::Float, 512, 32, 48>,
            GaplessConfig<21, Approach::unused, Datatype::Float, 512, 32, 50>,
            GaplessConfig<21, Approach::unused, Datatype::Float, 512, 32, 52>,
            GaplessConfig<21, Approach::unused, Datatype::Float, 512, 32, 54>,
            GaplessConfig<21, Approach::unused, Datatype::Float, 512, 32, 56>,
            GaplessConfig<21, Approach::unused, Datatype::Float, 512, 32, 58>,
            GaplessConfig<21, Approach::unused, Datatype::Float, 512, 32, 60>,
            GaplessConfig<21, Approach::unused, Datatype::Float, 512, 32, 62>,
            GaplessConfig<21, Approach::unused, Datatype::Float, 512, 32, 64>,
            >;

            // static constexpr auto getPerformanceHints(){
            //     constexpr auto result = std::array{724,1133,1548,1953,2367,2766,3181,3593,3976,4354,4754,5199,
            //         5589,5952,6375,6760,7038,6836,7451,7427,7988,8128,8861,8669,9281,9266,9861,9897,10479,10331,
            //         10935,11524,11858,12107,12126,12115,12420,12499,12645,12333,12761,12527,13294,13105,13240,
            //         12898,13851,13195,13493,13528,13310,13319,13521,13596,13544,13538,13714,13785,14809,14694,
            //         14030,13802,13815,14131,14388,14457,14175,14013,14424,14293,14657,14611,14718,14639,14975,
            //         15017,14938,14739,14692,13943,14350,14581,14582,14620,14790,15023,15119,15164,15230,15255,
            //         15680,15336,15401,15650,15515,};

            //     static_assert(result.size() == std::tuple_size_v<type>);

            //     return result;
            // };

            static constexpr bool hasPerformanceHints(){
                return false;
            }
        };
    
        struct MultiTileConfigs{
            using type = std::tuple<
                GaplessConfig<21, Approach::unused, Datatype::Float, 512, 8, 52>,
                GaplessConfig<21, Approach::unused, Datatype::Float, 512, 8, 54>,
                GaplessConfig<21, Approach::unused, Datatype::Float, 512, 8, 56>,
                GaplessConfig<21, Approach::unused, Datatype::Float, 512, 8, 58>,
                GaplessConfig<21, Approach::unused, Datatype::Float, 512, 8, 60>,
                GaplessConfig<21, Approach::unused, Datatype::Float, 512, 8, 62>,
                GaplessConfig<21, Approach::unused, Datatype::Float, 512, 8, 64>,
                GaplessConfig<21, Approach::unused, Datatype::Float, 256, 16, 34>,
                GaplessConfig<21, Approach::unused, Datatype::Float, 256, 16, 36>,
                GaplessConfig<21, Approach::unused, Datatype::Float, 256, 16, 38>,
                GaplessConfig<21, Approach::unused, Datatype::Float, 512, 32, 20>,
            >;


            // static constexpr auto getPerformanceHints(){
            //     constexpr auto result = std::array

            //     static_assert(result.size() == std::tuple_size_v<type>);

            //     return result;
            // };

            static constexpr bool hasPerformanceHints(){
                return false;
            }
        };
    
                    
        return std::make_unique<MultiConfigScoreOnlyAligner<SingleTileConfigs, MultiTileConfigs>>(AlignmentAlgorithmE::Gapless);
    }
#endif

    std::unique_ptr<MultiConfigScoreOnlyAlignerInterface> make_multiconfig_gapless_aligner_sm120_8bit_if_available_else_16bit(){


        #if defined(__CUDACC__) && !(__CUDACC_VER_MAJOR__ < 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ < 2))

        return make_multiconfig_gapless_aligner_sm120_8bit();

        #else

        return make_multiconfig_gapless_aligner_sm120_16bit();

        #endif
    }




#if 0
    std::unique_ptr<MultiConfigScoreOnlyAlignerInterface> make_multiconfig_sw_aligner_sm120_32bit(){
        struct SingleTileConfigs{
            using type = std::tuple<
                SWConfig<21, Approach::unused, Datatype::Float, 256, 8, 16>,
                SWConfig<21, Approach::unused, Datatype::Float, 256, 16, 16>,
            >;

            static constexpr bool hasPerformanceHints(){
                return false;
            }
        };
    
        struct MultiTileConfigs{
            using type = std::tuple<
                SWConfig<21, Approach::unused, Datatype::Float, 256, 16, 16>
            >;

            static constexpr bool hasPerformanceHints(){
                return false;
            }
        };
    
                    
        return std::make_unique<MultiConfigScoreOnlyAligner<SingleTileConfigs, MultiTileConfigs>>(AlignmentAlgorithmE::SW);
    }
#endif


LIBMARV_NAMESPACE_END