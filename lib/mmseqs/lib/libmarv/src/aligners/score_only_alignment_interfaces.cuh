#ifndef LIBMARV_SCOREONLY_ALIGNMENT_INTERFACES_CUH
#define LIBMARV_SCOREONLY_ALIGNMENT_INTERFACES_CUH


#include "../pssm.cuh"
#include "../util.cuh"
#include "../cuda_errorcheck.cuh"
#include "../alignment_algorithms/alignment_interface_data.cuh"
#include "../config.hpp"
#include "../offset_iterator.cuh"

#include <map>
#include <optional>
#include <cuda/std/mdspan>

#include "../namespace.hpp"
LIBMARV_NAMESPACE_BEGIN

    struct ScoreOnlyAlignmentInterface{
        virtual bool isSingleTile(const OneToAllInputDataPSSM& inputData) const = 0;

        virtual int getTileSize() const = 0;

        virtual AlignmentAlgorithmE getAlignmentAlgorithm() const = 0;

        virtual Datatype getScoreDatatype() const = 0;

        virtual void makeGpuPssm(
            GpuConvertedPSSM& result,
            const cuda::std::mdspan<std::int8_t, cuda::std::dextents<int,2>> hostPssmView,
            cudaStream_t stream
        ) = 0;

        virtual size_t getMinimumSuggestedTempBytes_multiTile(const OneToAllInputDataPSSM& inputData) const = 0;

        virtual void scoreOnly_singleTile(
            int* const devAlignmentScores,
            const OneToAllInputDataPSSM& inputData,
            const GpuConvertedPSSM& strided_PSSM,
            GapScoreArgs gapArgs,
            cudaStream_t stream
        ) = 0;

        virtual void scoreOnly_multiTile(
            char* d_temp,
            size_t tempBytes,
            int* const devAlignmentScores,
            const OneToAllInputDataPSSM& inputData,
            const GpuConvertedPSSM& strided_PSSM,
            GapScoreArgs gapArgs,
            cudaStream_t stream
        ) = 0;
    };

    struct MultiConfigScoreOnlyAlignerInterface{
        virtual bool isSingleTile(const OneToAllInputDataPSSM& inputData) const = 0;

        virtual int getTileSize(const OneToAllInputDataPSSM& inputData) const = 0;

        virtual AlignmentAlgorithmE getAlignmentAlgorithm() const = 0;

        virtual Datatype getScoreDatatype(int queryLength) const = 0;

        virtual void makeGpuPssm(
            GpuConvertedPSSM& result,
            const cuda::std::mdspan<std::int8_t, cuda::std::dextents<int,2>> hostPssmView,
            cudaStream_t stream
        ) = 0;

        virtual size_t getMinimumSuggestedTempBytes_multiTile(const OneToAllInputDataPSSM& inputData) const = 0;

       
        virtual void scoreOnly_singleTile(
            int* const devAlignmentScores,
            const OneToAllInputDataPSSM& inputData,
            const GpuConvertedPSSM& strided_PSSM,
            GapScoreArgs gapArgs,
            cudaStream_t stream
        ) = 0;

        virtual void scoreOnly_multiTile(
            char* d_temp,
            size_t tempBytes,
            int* const devAlignmentScores,
            const OneToAllInputDataPSSM& inputData,
            const GpuConvertedPSSM& strided_PSSM,
            GapScoreArgs gapArgs,
            cudaStream_t stream
        ) = 0;
    };
    


LIBMARV_NAMESPACE_END


#endif //LIBMARV_SCOREONLY_ALIGNMENT_INTERFACES_CUH