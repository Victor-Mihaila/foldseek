#ifndef LIBMARV_ALIGNMENT_INTERFACE_DATA_CUH
#define LIBMARV_ALIGNMENT_INTERFACE_DATA_CUH

#include "../offset_iterator.cuh"
#include "../util.cuh"
#include <cstdint>

#include "../namespace.hpp"
LIBMARV_NAMESPACE_BEGIN


struct OneToAllInputDataPSSM{
    public:
    
        OneToAllInputDataPSSM() = default;
        
        __host__ __device__
        int getNumAlignments() const{
            return numAlignments;
        }
    
        __host__ __device__
        const std::int8_t* getSubject(ReferenceIdT i) const{
            const auto index = indexIndirection[i];
            return d_subjects + d_subjectOffsets[index];
        }
    
        __host__ __device__
        SequenceLengthT getSubjectLength(ReferenceIdT i) const{
            const auto index = indexIndirection[i];
            return d_subjectLengths[index];
        }
    
        __host__ __device__
        SequenceLengthT getQueryLength() const{
            return queryLength;
        }
    
        __host__ __device__
        int getMaximumSubjectLength() const{
            return maximumSubjectLength;
        }   
    
    public: // API does not care about those
        const std::int8_t* d_subjects{};
        ReadOnlyCustomOffsetIterator<size_t*> d_subjectOffsets{};
        const SequenceLengthT* d_subjectLengths{};
        SequenceLengthT queryLength{};
        int numAlignments{};
        int maximumSubjectLength{};
        PositionsIterator indexIndirection = PositionsIterator::fromCountingIterator(0);
    };

    enum class AlignmentAlgorithmE{
        Gapless,
        SW
    };

    struct GapScoreArgs{
        int gapopenscore;
        int gapextendscore;
    };

    template<
        int alphabetSize_,
        Approach approach_,
        Datatype datatype_,
        int blocksize_,
        int groupsize_,
        int numRegs_
    >
    struct GaplessConfig{
        static constexpr AlignmentAlgorithmE algorithm = AlignmentAlgorithmE::Gapless;

        static constexpr int alphabetSize = alphabetSize_;
        static constexpr Approach approach = approach_;
        static constexpr Datatype datatype = datatype_;
        static constexpr int blocksize = blocksize_;
        static constexpr int groupsize = groupsize_;
        static constexpr int numRegs = numRegs_;
    };

    template<
        int alphabetSize_,
        Approach approach_,
        Datatype datatype_,
        int blocksize_,
        int groupsize_,
        int numRegs_
    >
    struct SWConfig{
        static constexpr AlignmentAlgorithmE algorithm = AlignmentAlgorithmE::SW;

        static constexpr int alphabetSize = alphabetSize_;
        static constexpr Approach approach = approach_;
        static constexpr Datatype datatype = datatype_;
        static constexpr int blocksize = blocksize_;
        static constexpr int groupsize = groupsize_;
        static constexpr int numRegs = numRegs_;
    };

LIBMARV_NAMESPACE_END


#endif //LIBMARV_ALIGNMENT_INTERFACE_DATA_CUH