#ifndef LIBMARV_SEQUENCE_MASKING_CUH
#define LIBMARV_SEQUENCE_MASKING_CUH

#if defined(__CUDACC__)
#include <cub/cub.cuh>
#else
#include <hipcub/hipcub.hpp>
namespace cub = hipcub;
#endif

#include <thrust/device_vector.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/gather.h>
#include <thrust/binary_search.h>

#include <cuda/functional>
#include <cuda/cmath>
#include <cuda/std/type_traits>

#include "cuda_errorcheck.cuh"
#include "util.cuh"

#if defined(__CUDACC__)
#include <cooperative_groups.h>
#include <cooperative_groups/scan.h>
#include <cooperative_groups/reduce.h>
#include <cooperative_groups/memcpy_async.h>
#else
#include <hip/hip_cooperative_groups.h>
#endif

#include "namespace.hpp"
LIBMARV_NAMESPACE_WITH_NESTING_BEGIN



template<int blocksize>
__global__
void setPaddingBytesKernel(
    char* d_data,
    const int* d_lengths,
    const size_t* d_offsets,
    int numSegments,
    char paddingLetter
){
    auto group = cooperative_groups::tiled_partition<4>(cooperative_groups::this_thread_block());
    constexpr int numGroupsInBlock = blocksize / 4;
    const int numGroupsInGrid = numGroupsInBlock * gridDim.x;
    const int groupIdInBlock = group.meta_group_rank();
    const int groupIdInGrid = (threadIdx.x + blockIdx.x * blockDim.x) / 4;

    for(int s = groupIdInGrid; s < numSegments; s += numGroupsInGrid){
        const size_t offset = d_offsets[s];
        const int length = d_lengths[s];
        const int paddedLength = cuda::ceil_div(length, 4) * 4;

        if(length + group.thread_rank() < paddedLength){
            d_data[offset + length + group.thread_rank()] = paddingLetter;
        }
    }
}



/*
    Requirements:

*/
template<class InputDataIterator, class InputOffsetIterator>
void maskSequences_rle_rld_impl(
    char* d_outputData,
    InputDataIterator d_inputData, //returns the letter (char) to consider for masking. each letter must be in range [0, 127]
    const int* d_inputLengths, //size numInputSequences
    InputOffsetIterator d_inputOffsets, //size numInputSequences
    size_t totalInputElements,
    int numInputSequences,
    int maskingThreshold, // replace letters in all repeats >= maskingThreshold by masking maskingLetter
    char maskingLetter,
    char paddingLetter,
    cudaStream_t stream
){
    using LetterIdTuple = cuda::std::tuple<char, uint16_t>;
    auto inputDataWithSegmentId = thrust::make_transform_iterator(
        thrust::make_counting_iterator<size_t>(0),
        cuda::proclaim_return_type<LetterIdTuple>([=] __device__ (size_t index){
            char letter = d_inputData[index];

            auto upIter = thrust::upper_bound(thrust::seq, d_inputOffsets, d_inputOffsets + numInputSequences, index);
            int segmentId = numInputSequences-1;
            if(upIter != d_inputOffsets + numInputSequences){
                segmentId = cuda::std::distance(d_inputOffsets, upIter) - 1;
            }
            return LetterIdTuple(letter, segmentId % 65536);
        })
    );

    // thrust::device_vector<LetterIdTuple> foo(inputDataWithSegmentId, inputDataWithSegmentId + totalInputElements);
    // std::cout << "foo\n";
    // for(int i = 0; i< totalInputElements; i++){
    //     LetterIdTuple t = foo[i];
    //     std::cout<< "(" << cuda::std::get<0>(t) << "," << cuda::std::get<1>(t) << ")" << " ";
    // }
    // std::cout << "\n";

    int numRuns;


    // thrust::device_vector<int, thrust_async_allocator<int>> d_numRuns(1, thrust_async_allocator<int>(stream));
    // //thrust::device_vector<char, thrust_async_allocator<char>> d_runValues(totalInputElements, thrust_async_allocator<char>(stream));

    // thrust::device_vector<LetterIdTuple, thrust_async_allocator<LetterIdTuple>> d_runValues_tuple(totalInputElements, thrust_async_allocator<LetterIdTuple>(stream));
    // thrust::device_vector<int, thrust_async_allocator<int>> d_runLengths(totalInputElements, thrust_async_allocator<int>(stream));


    int* d_numRuns_ptr; 
    CUDACHECK(cudaMallocAsync(&d_numRuns_ptr, sizeof(int) * 1, stream));
    // LetterIdTuple* d_runValues_tuple_ptr; 
    // CUDACHECK(cudaMallocAsync(&d_runValues_tuple_ptr, sizeof(LetterIdTuple) * totalInputElements, stream));
    char* d_runValues_ptr; 
    CUDACHECK(cudaMallocAsync(&d_runValues_ptr, sizeof(char) * totalInputElements, stream));
    uint16_t* d_runValues_segmentIds; 
    CUDACHECK(cudaMallocAsync(&d_runValues_segmentIds, sizeof(uint16_t) * totalInputElements, stream));
    int* d_runLengths_ptr; 
    CUDACHECK(cudaMallocAsync(&d_runLengths_ptr, sizeof(int) * totalInputElements, stream));


    auto rle_output = thrust::make_zip_iterator(
        d_runValues_ptr,
        d_runValues_segmentIds
    );


    // auto d_nonTupleRunValues = thrust::make_transform_output_iterator(
    //     d_runValues.begin(),
    //     cuda::proclaim_return_type<char>([] __device__ (LetterIdTuple tup){
    //         return cuda::std::get<0>(tup);
    //     })
    // );

    size_t cubBytes = 0;
    CUDACHECK(cub::DeviceRunLengthEncode::Encode(
        nullptr,
        cubBytes,
        inputDataWithSegmentId,
        rle_output,
        d_runLengths_ptr,
        d_numRuns_ptr,
        totalInputElements,
        stream
    ));

    char* d_temp; cudaMallocAsync(&d_temp, cubBytes, stream);

    CUDACHECK(cub::DeviceRunLengthEncode::Encode(
        d_temp,
        cubBytes,
        inputDataWithSegmentId,
        rle_output,
        d_runLengths_ptr,
        d_numRuns_ptr,
        totalInputElements,
        stream
    ));
    CUDACHECK(cudaFreeAsync(d_temp, stream));

    CUDACHECK(cudaMemcpyAsync(&numRuns, d_numRuns_ptr, sizeof(int), cudaMemcpyDeviceToHost, stream));
    CUDACHECK(cudaStreamSynchronize(stream));
    std::cout << "maskSequences: totalInputElements " << totalInputElements << ", numRuns " << numRuns << "\n";

    // std::cout << "d_runLengths\n";
    // for(int i = 0; i < numRuns; i++){
    //     std::cout << d_runLengths[i] << " ";
    // }
    // std::cout << "\n";

    #if 1

    //run-length decode

    thrust::inclusive_scan(
        thrust::cuda::par_nosync(thrust_async_allocator<char>(stream)).on(stream),
        d_runLengths_ptr,
        d_runLengths_ptr + numRuns,
        d_runLengths_ptr
    );

    thrust::transform(
        thrust::cuda::par_nosync(thrust_async_allocator<char>(stream)).on(stream),
        thrust::make_counting_iterator<size_t>(1),
        thrust::make_counting_iterator<size_t>(1) + totalInputElements,
        d_outputData,
        cuda::proclaim_copyable_arguments(
            [
                runLengthInclPS = d_runLengths_ptr, 
                d_runValues_ptr,
                numRuns,
                maskingThreshold,
                maskingLetter
            ] __device__ (size_t index){
                auto it = thrust::lower_bound(thrust::seq, runLengthInclPS, runLengthInclPS + numRuns, index);
                const int rle_segment = cuda::std::distance(runLengthInclPS, it);
                const int rle_length = rle_segment == 0 ? runLengthInclPS[0] : runLengthInclPS[rle_segment] - runLengthInclPS[rle_segment-1];
                
                if(rle_length >= maskingThreshold){
                    return maskingLetter;
                }else{
                    return d_runValues_ptr[rle_segment];
                }
            }
        )
    );


    #else

        //apply masking letter to runs >= maskingThreshold
        thrust::scatter_if(
            thrust::cuda::par_nosync.on(stream),
            thrust::make_constant_iterator(maskingLetter),
            thrust::make_constant_iterator(maskingLetter) + numRuns,
            thrust::make_counting_iterator(0),
            d_runLengths_ptr,
            d_runValues_ptr,
            // thrust::make_transform_output_iterator(
            //     d_runValues_tuple_ptr,
            //     cuda::proclaim_return_type<LetterIdTuple>([] __device__ (char letter){
            //         return LetterIdTuple(letter, 0);
            //     })
            // ),
            [=] __device__ (int runlength){
                return runlength >= maskingThreshold;
            }
        );


        //run-length decode

        thrust::inclusive_scan(
            thrust::cuda::par_nosync(thrust_async_allocator<char>(stream)).on(stream),
            d_runLengths_ptr,
            d_runLengths_ptr + numRuns,
            d_runLengths_ptr
        );

        auto srcLocations = thrust::make_transform_iterator(
            thrust::counting_iterator<int>(1),
            cuda::proclaim_return_type<int>([runLengthInclPS = d_runLengths_ptr, numRuns] __device__ (int index){
                auto it = thrust::lower_bound(thrust::seq, runLengthInclPS, runLengthInclPS + numRuns, index);
                return int(cuda::std::distance(runLengthInclPS, it));
            })
        );

        thrust::gather(
            thrust::cuda::par_nosync.on(stream),
            srcLocations,
            srcLocations + totalInputElements,
            d_runValues_ptr,
            // thrust::make_transform_iterator(
            //     d_runValues_tuple_ptr,
            //     cuda::proclaim_return_type<char>([] __device__ (LetterIdTuple tup){
            //         return cuda::std::get<0>(tup);
            //     })
            // ),    
            d_outputData
        );

    #endif

setPaddingBytesKernel<128><<<cuda::ceil_div(numInputSequences, 128 / 4), 128, 0, stream>>>(
        d_outputData,
        d_inputLengths,
        d_inputOffsets,
        numInputSequences,
        paddingLetter
    );
    CUDACHECKASYNC;


    CUDACHECK(cudaFreeAsync(d_runLengths_ptr, stream));
    // CUDACHECK(cudaFreeAsync(d_runValues_tuple_ptr, stream));
    CUDACHECK(cudaFreeAsync(d_runValues_ptr, stream));
    CUDACHECK(cudaFreeAsync(d_runValues_segmentIds, stream));
    CUDACHECK(cudaFreeAsync(d_numRuns_ptr, stream));
}




/*
    Requirements:

*/
template<class InputDataIterator, class InputOffsetIterator>
void maskSequences_simple_for_each_impl(
    char* d_outputData,
    InputDataIterator d_inputData, //returns the letter (char) to consider for masking. each letter must be in range [0, 127]
    const int* d_inputLengths, //size numInputSequences
    InputOffsetIterator d_inputOffsets, //size numInputSequences
    size_t totalInputElements,
    int numInputSequences,
    int maskingThreshold, // replace letters in all repeats >= maskingThreshold by masking maskingLetter
    char maskingLetter,
    char paddingLetter,
    cudaStream_t stream
){
    using LetterIdTuple = cuda::std::tuple<char, int>;
    auto inputDataWithSegmentId = thrust::make_transform_iterator(
        thrust::make_counting_iterator<size_t>(0),
        cuda::proclaim_return_type<LetterIdTuple>([=] __device__ (size_t index){
            char letter = d_inputData[index];

            auto upIter = thrust::upper_bound(thrust::seq, d_inputOffsets, d_inputOffsets + numInputSequences, index);
            int segmentId = numInputSequences-1;
            if(upIter != d_inputOffsets + numInputSequences){
                segmentId = cuda::std::distance(d_inputOffsets, upIter) - 1;
            }
            return LetterIdTuple(letter, segmentId);
        })
    );

    thrust::for_each(
        thrust::cuda::par_nosync(thrust_async_allocator<char>(stream)).on(stream),
        thrust::make_counting_iterator<int64_t>(0),
        thrust::make_counting_iterator<int64_t>(0) + totalInputElements,
        [
            maskingThreshold,
            maskingLetter,
            inputDataWithSegmentId,
            totalInputElements,
            d_outputData
        ]__device__ (int64_t index){
            const LetterIdTuple myData = inputDataWithSegmentId[index];

            int countLeft = 0;
            int countRight = 0;

            //check to the left
            for(int64_t i = index - 1; i > index - maskingThreshold; i--){
                if(i < 0){
                    break;
                }
                auto other = inputDataWithSegmentId[i];
                if(other == myData){
                    countLeft++;
                }else{
                    break;
                }
            }

            //check to the right
            for(int64_t i = index + 1; i < index + maskingThreshold; i++){
                if(i >= totalInputElements){
                    break;
                }
                auto other = inputDataWithSegmentId[i];
                if(other == myData){
                    countRight++;
                }else{
                    break;
                }
            }

            int count = countLeft + 1 + countRight;
            if(count >= maskingThreshold){
                d_outputData[index] = maskingLetter;
            }else{
                d_outputData[index] = cuda::std::get<0>(myData);
            }
        }
    );


    setPaddingBytesKernel<128><<<cuda::ceil_div(numInputSequences, 128 / 4), 128, 0, stream>>>(
        d_outputData,
        d_inputLengths,
        d_inputOffsets,
        numInputSequences,
        paddingLetter
    );
    CUDACHECKASYNC;

}



template<int maximumMaskingThreshold, class InputDataIterator, class SegmentIdIterator>
__global__
void maskingKernel(
    char* d_outputData,
    InputDataIterator d_inputData,
    SegmentIdIterator d_segmentIds,
    int64_t totalInputElements,
    int maskingThreshold,
    char maskingLetter
){

    using LetterIdTuple = cuda::std::tuple<char, int>;
    constexpr int64_t maxChunkSize = 1024;
    constexpr int smemElements = maxChunkSize + 2 * (maximumMaskingThreshold-1);
    __shared__ char smem_letters[smemElements];
    __shared__ int smem_segmentIds[smemElements];

    const int64_t numChunks = cuda::ceil_div(totalInputElements, maxChunkSize);

    for(int64_t chunkId = blockIdx.x; chunkId < numChunks; chunkId += gridDim.x){
        const int64_t chunkBegin = chunkId * maxChunkSize;
        const int64_t chunkEnd_excl = cuda::std::min(totalInputElements, (chunkId+1) * maxChunkSize);
        // const int chunkSize = chunkEnd_excl - chunkBegin;

        const int64_t loadBegin = cuda::std::max(int64_t(0), chunkBegin - (maximumMaskingThreshold-1));
        const int64_t loadEnd_excl = cuda::std::min(totalInputElements, chunkEnd_excl + (maximumMaskingThreshold-1));
        const int elementsToLoad = loadEnd_excl - loadBegin;

        // if(threadIdx.x == 0){
            
        //     if(elementsToLoad > smemElements){
        //         printf("loadBegin %ld, chunkBegin %lu, chunkEnd_excl %lu, loadEnd_excl %ld, elementsToLoad %d, smemElements %d\n", 
        //             loadBegin, chunkBegin, chunkEnd_excl, loadEnd_excl, elementsToLoad, smemElements);
        //     }
        //     assert(elementsToLoad <= smemElements);
        // }

        //load chunk to shared memory
        __syncthreads();
        for(int64_t i = loadBegin + threadIdx.x; i < loadEnd_excl; i += blockDim.x){
            smem_letters[i - loadBegin] = d_inputData[i];
            smem_segmentIds[i - loadBegin] = d_segmentIds[i];
        }
        __syncthreads();

        //process chunk
        for(int64_t element = chunkBegin + threadIdx.x; element < chunkEnd_excl; element += blockDim.x){
            const int indexInSmem = element - loadBegin;
            const char myLetter = smem_letters[indexInSmem];
            const int mySegmentId = smem_segmentIds[indexInSmem];

            int count = 1; //account for my letter
            //check to the left
            for(int i = indexInSmem - 1; i > indexInSmem - maskingThreshold; i--){
                if(i < 0){
                    break;
                }
                const char otherLetter = smem_letters[i];
                const int otherSegmentId = smem_segmentIds[i];
                if(myLetter == otherLetter && mySegmentId == otherSegmentId){
                    count++;
                }else{
                    break;
                }
            }

            //check to the right
            for(int64_t i = indexInSmem + 1; i < indexInSmem + maskingThreshold; i++){
                if(i >= elementsToLoad){
                    break;
                }
                const char otherLetter = smem_letters[i];
                const int otherSegmentId = smem_segmentIds[i];
                if(myLetter == otherLetter && mySegmentId == otherSegmentId){
                    count++;
                }else{
                    break;
                }
            }


            if(count >= maskingThreshold){
                d_outputData[element] = maskingLetter;
            }else{
                d_outputData[element] = myLetter;
            }
        }

    }
}

/*
    Requirements:

*/
template<int maximumMaskingThreshold, class InputDataIterator, class InputOffsetIterator>
void maskSequences_custom_kernel_impl(
    char* d_outputData,
    InputDataIterator d_inputData, //returns the letter (char) to consider for masking. each letter must be in range [0, 127]
    const int* d_inputLengths, //size numInputSequences
    InputOffsetIterator d_inputOffsets, //size numInputSequences
    size_t totalInputElements,
    int numInputSequences,
    int maskingThreshold, // replace letters in all repeats >= maskingThreshold by masking maskingLetter
    char maskingLetter,
    char paddingLetter,
    cudaStream_t stream
){
    assert(maskingThreshold <= maximumMaskingThreshold);

    auto d_segmentIds = thrust::make_transform_iterator(
        thrust::make_counting_iterator<size_t>(0),
        cuda::proclaim_return_type<int>([=] __device__ (size_t index){
            auto upIter = thrust::upper_bound(thrust::seq, d_inputOffsets, d_inputOffsets + numInputSequences, index);
            int segmentId = numInputSequences-1;
            if(upIter != d_inputOffsets + numInputSequences){
                segmentId = cuda::std::distance(d_inputOffsets, upIter) - 1;
            }
            return segmentId;
        })
    );

    int deviceId;
    int numSMs;
    int maxBlocksPerSM = 0;
    constexpr int blocksize = 128;
    auto kernel = maskingKernel<maximumMaskingThreshold, InputDataIterator, decltype(d_segmentIds)>;

    CUDACHECK(cudaGetDevice(&deviceId));
    CUDACHECK(cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, deviceId));
    CUDACHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &maxBlocksPerSM,
        kernel,
        blocksize, 
        0
    ));

    kernel<<<numSMs*maxBlocksPerSM, blocksize, 0, stream>>>(
        d_outputData,
        d_inputData,
        d_segmentIds,
        totalInputElements,
        maskingThreshold,
        maskingLetter
    );
    CUDACHECKASYNC;

    setPaddingBytesKernel<128><<<cuda::ceil_div(numInputSequences, 128 / 4), 128, 0, stream>>>(
        d_outputData,
        d_inputLengths,
        d_inputOffsets,
        numInputSequences,
        paddingLetter
    );
    CUDACHECKASYNC;

}




    
template<int maximumMaskingThreshold, int maximumChunkSize, class InputDataIterator, class SegmentIdIterator>
__global__
void maskingKernel2(
    char* d_outputData,
    InputDataIterator d_inputData,
    SegmentIdIterator d_segmentIds,
    int64_t totalInputElements,
    int maskingThreshold,
    char maskingLetter
){

    using LetterIdTuple = cuda::std::tuple<char, int>;
    constexpr int64_t maxChunkSize = maximumChunkSize;
    constexpr int smemElements = maxChunkSize + 2 * (maximumMaskingThreshold-1);

    struct Pair{
        char letter;
        uint16_t id;
    };
    __shared__ Pair smem_pairs[smemElements];

    const int64_t numChunks = cuda::ceil_div(totalInputElements, maxChunkSize);

    for(int64_t chunkId = blockIdx.x; chunkId < numChunks; chunkId += gridDim.x){
        const int64_t chunkBegin = chunkId * maxChunkSize;
        const int64_t chunkEnd_excl = cuda::std::min(totalInputElements, (chunkId+1) * maxChunkSize);
        // const int chunkSize = chunkEnd_excl - chunkBegin;

        const int64_t loadBegin = cuda::std::max(int64_t(0), chunkBegin - (maximumMaskingThreshold-1));
        const int64_t loadEnd_excl = cuda::std::min(totalInputElements, chunkEnd_excl + (maximumMaskingThreshold-1));
        const int elementsToLoad = loadEnd_excl - loadBegin;

        // if(threadIdx.x == 0){
            
        //     if(elementsToLoad > smemElements){
        //         printf("loadBegin %ld, chunkBegin %lu, chunkEnd_excl %lu, loadEnd_excl %ld, elementsToLoad %d, smemElements %d\n", 
        //             loadBegin, chunkBegin, chunkEnd_excl, loadEnd_excl, elementsToLoad, smemElements);
        //     }
        //     assert(elementsToLoad <= smemElements);
        // }

        //load chunk to shared memory
        __syncthreads();
        for(int64_t i = loadBegin + threadIdx.x; i < loadEnd_excl; i += blockDim.x){
            Pair p;
            p.letter = d_inputData[i];
            // p.id = d_segmentIds[i] % (65536);
            p.id = d_segmentIds[i];
            smem_pairs[i - loadBegin] = p;
        }
        __syncthreads();

        //process chunk
        for(int64_t element = chunkBegin + threadIdx.x; element < chunkEnd_excl; element += blockDim.x){
            const int indexInSmem = element - loadBegin;
            const Pair myPair = smem_pairs[indexInSmem];
            const char myLetter = myPair.letter;
            const int mySegmentId = myPair.id;

            int count = 1; //account for my letter
            //check to the left
            for(int i = indexInSmem - 1; i > indexInSmem - maskingThreshold; i--){
                if(i < 0){
                    break;
                }
                const Pair otherPair = smem_pairs[i];
                const char otherLetter = otherPair.letter;
                const int otherSegmentId = otherPair.id;
                if(myLetter == otherLetter && mySegmentId == otherSegmentId){
                    count++;
                }else{
                    break;
                }
            }

            //check to the right
            for(int64_t i = indexInSmem + 1; i < indexInSmem + maskingThreshold; i++){
                if(count >= maskingThreshold){
                    break;
                }
                if(i >= elementsToLoad){
                    break;
                }
                const Pair otherPair = smem_pairs[i];
                const char otherLetter = otherPair.letter;
                const int otherSegmentId = otherPair.id;
                if(myLetter == otherLetter && mySegmentId == otherSegmentId){
                    count++;
                }else{
                    break;
                }
            }

            // if(element >= 736120 && element < 736130){
            //     printf("element %ld, letter %d, count %d\n", element, int(myLetter), count);
            // }


            if(count >= maskingThreshold){
                d_outputData[element] = maskingLetter;
            }else{
                d_outputData[element] = myLetter;
            }
        }

    }
}




template<int maximumMaskingThreshold, class InputDataIterator, class InputOffsetIterator>
void maskSequences_custom_kernel_impl2(
    char* d_outputData,
    InputDataIterator d_inputData, //returns the letter (char) to consider for masking. each letter must be in range [0, 127]
    const int* d_inputLengths, //size numInputSequences
    InputOffsetIterator d_inputOffsets, //size numInputSequences
    size_t totalInputElements,
    int numInputSequences,
    int maskingThreshold, // replace letters in all repeats >= maskingThreshold by masking maskingLetter
    char maskingLetter,
    char paddingLetter,
    cudaStream_t stream
){
    assert(maskingThreshold <= maximumMaskingThreshold);

    // auto d_segmentIds = thrust::make_transform_iterator(
    //     thrust::make_counting_iterator<size_t>(0),
    //     cuda::proclaim_return_type<int>([=] __device__ (size_t index){
    //         auto upIter = thrust::upper_bound(thrust::seq, d_inputOffsets, d_inputOffsets + numInputSequences, index);
    //         int segmentId = numInputSequences-1;
    //         if(upIter != d_inputOffsets + numInputSequences){
    //             segmentId = cuda::std::distance(d_inputOffsets, upIter) - 1;
    //         }
    //         return segmentId;
    //     })
    // );

    //we only need to distinguish segments in each subchunk in the kernel, ushort is sufficient and provides safe wrap-around for the scan.
    constexpr int maximumChunkSize = 1024;
    uint16_t* d_segmentIds; 
    CUDACHECK(cudaMallocAsync(&d_segmentIds, sizeof(uint16_t) * totalInputElements, stream));
    CUDACHECK(cudaMemsetAsync(d_segmentIds, 0, sizeof(uint16_t) * totalInputElements, stream));
    thrust::scatter(
        thrust::cuda::par_nosync.on(stream),
        thrust::make_constant_iterator<uint16_t>(1),
        thrust::make_constant_iterator<uint16_t>(1) + numInputSequences,
        d_inputOffsets,
        d_segmentIds
    );
    thrust::inclusive_scan(
        thrust::cuda::par_nosync(thrust_async_allocator<char>(stream)).on(stream),
        d_segmentIds,
        d_segmentIds + totalInputElements,
        d_segmentIds
    );

    int deviceId;
    int numSMs;
    int maxBlocksPerSM = 0;
    constexpr int blocksize = 128;
    auto kernel = maskingKernel2<maximumMaskingThreshold, maximumChunkSize, InputDataIterator, decltype(d_segmentIds)>;

    CUDACHECK(cudaGetDevice(&deviceId));
    CUDACHECK(cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, deviceId));
    CUDACHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &maxBlocksPerSM,
        kernel,
        blocksize, 
        0
    ));

    kernel<<<numSMs*maxBlocksPerSM, blocksize, 0, stream>>>(
        d_outputData,
        d_inputData,
        d_segmentIds,
        totalInputElements,
        maskingThreshold,
        maskingLetter
    );
    CUDACHECKASYNC;

    setPaddingBytesKernel<128><<<cuda::ceil_div(numInputSequences, 128 / 4), 128, 0, stream>>>(
        d_outputData,
        d_inputLengths,
        d_inputOffsets,
        numInputSequences,
        paddingLetter
    );
    CUDACHECKASYNC;

    CUDACHECK(cudaFreeAsync(d_segmentIds, stream));

}




template<int blocksize, class InputDataIterator, class SegmentIdIterator>
__global__
void maskingKernel3(
    char* d_outputData,
    InputDataIterator d_inputData,
    SegmentIdIterator d_segmentIds,
    int64_t totalInputElements,
    int maskingThreshold,
    char maskingLetter
){
    __builtin_assume(blocksize == blockDim.x);

    struct RunIdCountPair{
        int runId;
        int count;
    };
    struct RunIdCountPairScanOp{
        __device__
        RunIdCountPair operator()(const RunIdCountPair& left, const RunIdCountPair& right) const{
            RunIdCountPair result;
            result.runId = right.runId;
            if(left.runId == right.runId){
                result.count = right.count + left.count;
            }else{
                result.count = right.count;                
            }
            return result;
        }
    };

    constexpr uint32_t oob_element = 0xFFFFFFFF;



    constexpr int items_per_thread = 16;
    constexpr int items_per_block = blocksize * items_per_thread;

    using BlockExchange = cub::BlockExchange<uint32_t, blocksize, items_per_thread>;
    using BlockDiscontinuity = cub::BlockDiscontinuity<uint32_t, blocksize>;
    using BlockScanInt = cub::BlockScan<int, blocksize>;
    using BlockScanPair = cub::BlockScan<RunIdCountPair, blocksize>;
    

    __shared__ struct{
        typename BlockExchange::TempStorage temp_exchange;
        typename BlockDiscontinuity::TempStorage temp_discontinuity;
        typename BlockScanInt::TempStorage temp_scanInt;
        typename BlockScanPair::TempStorage temp_scanPair;
    } temp;

    __shared__ char smem_letters[items_per_block];
    __shared__ int smem_runLengths[items_per_block];



    /*
        Example for chunking: with 2048 items per block and maskingThreshold == 3, 2045 is the last index that could be 
        the start of a run to mask in the current chunk. index 2046 has to be checked by the next chunk. Thus, next chunk
        needs to start at index 2044 to check if 2046 is the end of a run to mask.
    */

    for(int64_t chunkBegin = blockIdx.x * (items_per_block - 2*(maskingThreshold - 1)) ; 
            chunkBegin < totalInputElements; 
            chunkBegin += gridDim.x * (items_per_block - 2*(maskingThreshold - 1))){

        const int64_t chunkEnd_excl = cuda::std::min(totalInputElements, chunkBegin + items_per_block);
        // const int64_t numInChunk = chunkEnd_excl - chunkBegin;

        uint32_t myElements[items_per_thread];        

        #pragma unroll
        for(int i = 0; i < items_per_thread; i++){
            const int loadIndex = chunkBegin + i * blockDim.x + threadIdx.x;
            if(loadIndex < chunkEnd_excl){
                uint32_t letter = d_inputData[loadIndex];
                //uint32_t id = d_segmentIds[loadIndex] % (65536);
                uint32_t id = d_segmentIds[loadIndex];
                uint32_t element = (letter << 16) | id;
                myElements[i] = element;
            }else{
                myElements[i] = oob_element;
            }
        }
        BlockExchange(temp.temp_exchange).StripedToBlocked(myElements, myElements);
        // __syncthreads();

        // if(threadIdx.x == 0){
        //     printf("elements\n");
        //     #pragma unroll
        //     for(int i = 0; i < items_per_thread; i++){
        //         printf("%u ", myElements[i]);
        //     }
        //     printf("\n");
        // }

        // const int printindex = 134088448;

        // #pragma unroll
        // for(int i = 0; i < items_per_thread; i++){
        //     if(chunkBegin + threadIdx.x * items_per_thread + i >= (printindex-5) && chunkBegin + threadIdx.x * items_per_thread + i <= (printindex+5)){
        //         printf("chunkId %ld, element %ld, %u\n", chunkId, chunkBegin + threadIdx.x * items_per_thread + i, myElements[i]);
        //     }
        // }


        int tailFlags[items_per_thread];
        BlockDiscontinuity(temp.temp_discontinuity).FlagTails(tailFlags, myElements, cuda::std::not_equal_to{});
        #pragma unroll
        for(int i = 0; i < items_per_thread; i++){
            if(chunkBegin + threadIdx.x * items_per_thread + i >= chunkEnd_excl){
                tailFlags[i] = 0;
            }
        }

        // if(threadIdx.x == 0){
        //     printf("tailFlags\n");
        //     #pragma unroll
        //     for(int i = 0; i < items_per_thread; i++){
        //         printf("%u ", tailFlags[i]);
        //     }
        //     printf("\n");
        // }

        // #pragma unroll
        // for(int i = 0; i < items_per_thread; i++){
        //     if(chunkBegin + threadIdx.x * items_per_thread + i >= (printindex-5) && chunkBegin + threadIdx.x * items_per_thread + i <= (printindex+5)){
        //         printf("chunkId %ld, tailFlags %ld, %u\n", chunkId, chunkBegin + threadIdx.x * items_per_thread + i, tailFlags[i]);
        //     }
        // }

        //exclusive sum to compute run ids marked by tail flags
        int prefixsum[items_per_thread];
        BlockScanInt(temp.temp_scanInt).ExclusiveSum(tailFlags, prefixsum);
        // __syncthreads();

        // if(threadIdx.x == 0){
        //     printf("prefixsum\n");
        //     #pragma unroll
        //     for(int i = 0; i < items_per_thread; i++){
        //         printf("%u ", prefixsum[i]);
        //     }
        //     printf("\n");
        // }

        // #pragma unroll
        // for(int i = 0; i < items_per_thread; i++){
        //     if(chunkBegin + threadIdx.x * items_per_thread + i >= (printindex-5) && chunkBegin + threadIdx.x * items_per_thread + i <= (printindex+5)){
        //         printf("chunkId %ld, prefixsum %ld, %u\n", chunkId, chunkBegin + threadIdx.x * items_per_thread + i, prefixsum[i]);
        //     }
        // }



        //inclusive scan to compute run lengths
        RunIdCountPair zippedprefixsum[items_per_thread];
        {
            RunIdCountPair zipped[items_per_thread];
            #pragma unroll
            for(int i = 0; i < items_per_thread; i++){
                zipped[i].count = 1;
                zipped[i].runId = prefixsum[i];
            }
            RunIdCountPairScanOp pairScanOp{};
            BlockScanPair(temp.temp_scanPair).InclusiveScan(zipped, zippedprefixsum, pairScanOp);
            // __syncthreads();

        }


        #pragma unroll
        for(int i = 0; i < items_per_thread; i++){
            if(tailFlags[i] == 1){
                smem_runLengths[prefixsum[i]] = zippedprefixsum[i].count;
            }
        }
        __syncthreads();

// if(warp.thread_rank() == 0) printf("E\n");
        int myRunLengths[items_per_thread];
        #pragma unroll
        for(int i = 0; i < items_per_thread; i++){
            myRunLengths[i] = smem_runLengths[zippedprefixsum[i].runId];
        }

        #pragma unroll
        for(int i = 0; i < items_per_thread; i++){
            if(chunkBegin + threadIdx.x * items_per_thread + i < chunkEnd_excl){
                char letter = (myElements[i] >> 16);
                if(myRunLengths[i] >= maskingThreshold){
                    letter = maskingLetter;
                }
                smem_letters[threadIdx.x * items_per_thread + i] = letter;
            }
        }
        __syncthreads();


// if(warp.thread_rank() == 0) printf("F\n");

        //write to output

        const bool isFirstChunk = chunkBegin == 0;
        const bool isLastChunk = chunkEnd_excl == totalInputElements;

        #pragma unroll
        for(int i = 0; i < items_per_thread; i++){
            const int index = i * blockDim.x + threadIdx.x;
            if(!isFirstChunk && !isLastChunk){
                if(index >= (maskingThreshold - 1)){
                    if(index < items_per_block - (maskingThreshold - 1) ){
                        d_outputData[chunkBegin + index] = smem_letters[index];

                        // if(chunkBegin + index == 2046){
                        //     printf("AAA chunkId %ld, write %c to %d. index %d, \n", chunkId, smem_letters[index], 2046, index);
                        // }
                    }
                }
            }else if(!isFirstChunk && isLastChunk){
                if(index >= (maskingThreshold - 1)){
                    if(chunkBegin + index < chunkEnd_excl){
                        d_outputData[chunkBegin + index] = smem_letters[index];

                        // if(chunkBegin + index == 2046){
                        //     printf("BBB chunkId %ld, write %c to %d. index %d, \n", chunkId, smem_letters[index], 2046, index);
                        // }
                    }
                }
            }else if(isFirstChunk && !isLastChunk){
                if(index < items_per_block - (maskingThreshold - 1) ){
                    d_outputData[chunkBegin + index] = smem_letters[index];

                    // if(chunkBegin + index == 2046){
                    //     printf("CCC chunkId %ld, write %c to %d. index %d, \n", chunkId, smem_letters[index], 2046, index);
                    // }
                }
            }else{
                //first and last
                if(chunkBegin + index < chunkEnd_excl){
                    d_outputData[chunkBegin + index] = smem_letters[index];

                    // if(chunkBegin + index == 2046){
                    //     printf("DDD chunkId %ld, write %c to %d. index %d, \n", chunkId, smem_letters[index], 2046, index);
                    // }
                }
            }
        }


        // #pragma unroll
        // for(int i = 0; i < items_per_thread; i++){
        //     const int index = i * blockDim.x + threadIdx.x;
        //     if(chunkBegin + index < chunkEnd_excl){
        //         if(chunkId == numChunks-1){
        //             if(chunkBegin + index == printindex){
        //                 printf("AAA write %c to %d\n", smem_letters[index], printindex);
        //                 }
        //             d_outputData[chunkBegin + index] = smem_letters[index];
        //         }else{
        //             if(index < items_per_block - (maskingThreshold - 1)){
        //                 if(chunkBegin + index == printindex){
        //                     printf("BBB chunkId %ld, write %c to %d. index %d, \n", chunkId, smem_letters[index], printindex, index);
        //                 }
        //                 d_outputData[chunkBegin + index] = smem_letters[index];
        //             }
        //         }
        //     }
        // }

        // #pragma unroll
        // for(int i = 0; i < items_per_thread; i++){
        //     const int index = i * blockDim.x + threadIdx.x;
        //     if(chunkBegin + index < chunkEnd_excl){
        //         if(chunkId == 0){
        //             if(chunkBegin + index == printindex){
        //                 printf("AAA write %c to %d\n", smem_letters[index], printindex);
        //                 }
        //             d_outputData[chunkBegin + index] = smem_letters[index];
        //         }else{
        //             //avoid overwriting results from previous overlapping chunk
        //             if(index >= (maskingThreshold - 1)){
        //                 if(chunkBegin + index == printindex){
        //                     printf("BBB chunkId %ld, write %c to %d. index %d, \n", chunkId, smem_letters[index], printindex, index);
        //                 }
        //                 d_outputData[chunkBegin + index] = smem_letters[index];
        //             }
        //         }
        //     }
        // }

// if(warp.thread_rank() == 0) printf("G\n");


        __syncthreads();
    }
}




template<class InputDataIterator, class InputOffsetIterator>
void maskSequences_custom_kernel_impl3(
    char* d_outputData,
    InputDataIterator d_inputData, //returns the letter (char) to consider for masking. each letter must be in range [0, 127]
    const int* d_inputLengths, //size numInputSequences
    InputOffsetIterator d_inputOffsets, //size numInputSequences
    size_t totalInputElements,
    int numInputSequences,
    int maskingThreshold, // replace letters in all repeats >= maskingThreshold by masking maskingLetter
    char maskingLetter,
    char paddingLetter,
    cudaStream_t stream
){


    // auto d_segmentIds = thrust::make_transform_iterator(
    //     thrust::make_counting_iterator<size_t>(0),
    //     cuda::proclaim_return_type<int>([=] __device__ (size_t index){
    //         auto upIter = thrust::upper_bound(thrust::seq, d_inputOffsets, d_inputOffsets + numInputSequences, index);
    //         int segmentId = numInputSequences-1;
    //         if(upIter != d_inputOffsets + numInputSequences){
    //             segmentId = cuda::std::distance(d_inputOffsets, upIter) - 1;
    //         }
    //         return segmentId;
    //     })
    // );

    //we only need to distinguish segments in each subchunk in the kernel, ushort is sufficient and provides safe wrap-around for the scan.

    uint16_t* d_segmentIds; 
    CUDACHECK(cudaMallocAsync(&d_segmentIds, sizeof(uint16_t) * totalInputElements, stream));
    CUDACHECK(cudaMemsetAsync(d_segmentIds, 0, sizeof(uint16_t) * totalInputElements, stream));
    thrust::scatter(
        thrust::cuda::par_nosync.on(stream),
        thrust::make_constant_iterator<uint16_t>(1),
        thrust::make_constant_iterator<uint16_t>(1) + numInputSequences,
        d_inputOffsets,
        d_segmentIds
    );
    thrust::inclusive_scan(
        thrust::cuda::par_nosync(thrust_async_allocator<char>(stream)).on(stream),
        d_segmentIds,
        d_segmentIds + totalInputElements,
        d_segmentIds
    );

    int deviceId;
    int numSMs;
    int maxBlocksPerSM = 0;
    constexpr int blocksize = 128;
    auto kernel = maskingKernel3<blocksize, InputDataIterator, decltype(d_segmentIds)>;

    CUDACHECK(cudaGetDevice(&deviceId));
    CUDACHECK(cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, deviceId));
    CUDACHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &maxBlocksPerSM,
        kernel,
        blocksize, 
        0
    ));

    kernel<<<numSMs*maxBlocksPerSM, blocksize, 0, stream>>>(
        d_outputData,
        d_inputData,
        d_segmentIds,
        totalInputElements,
        maskingThreshold,
        maskingLetter
    );
    CUDACHECKASYNC;

    setPaddingBytesKernel<128><<<cuda::ceil_div(numInputSequences, 128 / 4), 128, 0, stream>>>(
        d_outputData,
        d_inputLengths,
        d_inputOffsets,
        numInputSequences,
        paddingLetter
    );
    CUDACHECKASYNC;

    CUDACHECK(cudaFreeAsync(d_segmentIds, stream));

}




template<int blocksize, int items_per_thread, class InputDataIterator, class InputOffsetIterator>
__global__
void maskingKernel4_smem_rle_rld_1warpPerSegment(
    char* d_outputData,
    InputDataIterator d_inputData,
    const int* d_lengths,
    InputOffsetIterator d_offsets,
    int64_t totalInputElements,
    int numSegments,
    int maskingThreshold,
    char maskingLetter,
    char paddingLetter
){

    constexpr int items_per_warp = items_per_thread * 32;

    auto warp = cooperative_groups::tiled_partition<32>(cooperative_groups::this_thread_block());
    constexpr int numWarpsInBlock = blocksize / 32;
    const int numWarpsInGrid = numWarpsInBlock * gridDim.x;
    const int warpIdInBlock = warp.meta_group_rank();
    const int warpIdInGrid = (threadIdx.x + blockIdx.x * blockDim.x) / 32;

    constexpr char oob_element = cuda::std::numeric_limits<char>::max();

    struct RunIdCountPair{
        int runId;
        int count;
    };
    struct RunIdCountPairScanOp{
        __device__
        RunIdCountPair operator()(const RunIdCountPair& left, const RunIdCountPair& right) const{
            RunIdCountPair result;
            result.runId = right.runId;
            if(left.runId == right.runId){
                result.count = right.count + left.count;
            }else{
                result.count = right.count;                
            }
            return result;
        }
    };

    using WarpLoadChar = cub::WarpLoad<char, items_per_thread, cub::WARP_LOAD_TRANSPOSE>;
    using WarpLoadChar4 = cub::WarpLoad<char4, items_per_thread / 4, cub::WARP_LOAD_TRANSPOSE>;
    using WarpScanInt = cub::WarpScan<int>;
    using WarpScanPair = cub::WarpScan<RunIdCountPair>;

    __shared__ struct Temp{
        struct U{
            typename WarpLoadChar::TempStorage temp_warploadChar[numWarpsInBlock];
            //typename WarpLoadChar4::TempStorage temp_warploadChar4[numWarpsInBlock];
            // char4 loaded_letters4[numWarpsInBlock][items_per_warp/4];
            char masked_letters[numWarpsInBlock][items_per_warp];
        } u;
        typename WarpScanInt::TempStorage temp_warpScanInt[numWarpsInBlock];
        typename WarpScanPair::TempStorage temp_warpScanPair[numWarpsInBlock];
        int runOffsets[numWarpsInBlock][items_per_warp];
        char runValues[numWarpsInBlock][items_per_warp];
    } temp;



    for(int segmentId = warpIdInGrid; segmentId < numSegments; segmentId += numWarpsInGrid){
        const size_t segmentOffset = d_offsets[segmentId];
        const int segmentSize = d_lengths[segmentId];
        const auto segmentInputData = d_inputData + segmentOffset;
        char* const segmentOutputData = d_outputData + segmentOffset;

        const int numChunks = cuda::ceil_div(segmentSize, items_per_warp);

        char lastElementOfPreviousChunk = oob_element;
        int lastCountOfPreviousChunk = 0;
        int writtenOutputElements = 0;

        for(int chunkId = 0; chunkId < numChunks; chunkId++){
            const int chunkBegin = chunkId * items_per_warp;
            const int chunkEnd_excl = cuda::std::min(segmentSize, chunkBegin + items_per_warp);
            const int numInChunk = chunkEnd_excl - chunkBegin;

            // if(warpIdInBlock == 0 && warp.thread_rank() == 3){
            //     printf("segmentOffset %lu, chunkBegin %d, chunkEnd_excl %d, numInChunk %d\n", segmentOffset, chunkBegin, chunkEnd_excl, numInChunk);
            // }

            alignas(4) char items[items_per_thread];

            WarpLoadChar(temp.u.temp_warploadChar[warpIdInBlock]).Load(segmentInputData + chunkBegin, items, numInChunk, oob_element);

            // {
            //     for(int i = warp.thread_rank(); i < cuda::ceil_div(numInChunk,4); i += warp.size()){
            //         temp.u.loaded_letters4[warpIdInBlock][i] = ((const char4*)(segmentInputData + chunkBegin))[i];
            //     }
            //     warp.sync();
            //     #pragma unroll
            //     for(int i = 0; i < items_per_thread / 4; i++){
            //         if(warp.thread_rank() * (items_per_thread/4) + i < cuda::ceil_div(numInChunk,4)){
            //             char4 tmp = temp.u.loaded_letters4[warpIdInBlock][warp.thread_rank() * (items_per_thread/4)  + i];
            //             memcpy(&items[4*i], &tmp, 4);
            //         }else{
            //             unsigned int tmp = 0x7F7F7F7F;
            //             memcpy(&items[4*i], &tmp, 4);
            //         }
            //     }
            //     #pragma unroll
            //     for(int i = 0; i < items_per_thread; i++){
            //         if(warp.thread_rank() * (items_per_thread) + i >= numInChunk){
            //             items[i] = oob_element;
            //         }
            //     }
            // }

            // char4 items4[items_per_thread / 4];
            // WarpLoadChar4(temp.u.temp_warploadChar4[warpIdInBlock]).Load((const char4*)(segmentInputData + chunkBegin), items4, cuda::ceil_div(numInChunk,4), char4{127,127,127,127});
            // memcpy(&items[0], &items4[0], sizeof(char) * items_per_thread);

            warp.sync(); //sync to ensure smem union u can be reused

            // if(warpIdInBlock == 0 && warp.thread_rank() == 3){
            //     printf("items\n");
            //     #pragma unroll
            //     for(int i = 0; i < items_per_thread; i++){
            //         printf("%d ", int(items[i]));
            //     }
            //     printf("\n");
            // }

            int tailFlags[items_per_thread];
            {
                auto successoritem = warp.shfl_down(items[0], 1);
                if(warp.thread_rank() == warp.size()-1){
                    if(chunkEnd_excl < segmentSize){
                        successoritem = segmentInputData[chunkEnd_excl];
                    }else{
                        successoritem = oob_element;
                    }
                }
                #pragma unroll
                for(int i = 0; i < items_per_thread-1; i++){
                    tailFlags[i] = items[i] != items[i+1];
                }
                tailFlags[items_per_thread-1] = items[items_per_thread-1] != successoritem;
            }

            // if(warpIdInBlock == 0 && warp.thread_rank() == 10){
            //     printf("tailFlags\n");
            //     #pragma unroll
            //     for(int i = 0; i < items_per_thread; i++){
            //         printf("%d ", tailFlags[i]);
            //     }
            //     printf("\n");
            // }

            //exclusive scan of tail flags to compute num runs
            int prefixsum[items_per_thread];
            int warpNumRuns = 0;
            {
                prefixsum[0] = 0;
                #pragma unroll
                for(int i = 1; i < items_per_thread; i++){
                    prefixsum[i] = prefixsum[i-1] + tailFlags[i-1];
                }
                int inclTotal = prefixsum[items_per_thread-1] + tailFlags[items_per_thread-1];
                int predTotal = 0;
                WarpScanInt(temp.temp_warpScanInt[warpIdInBlock]).ExclusiveSum(inclTotal, predTotal, warpNumRuns);
                for(int i = 0; i < items_per_thread; i++){
                    prefixsum[i] += predTotal;
                }
            }

            // if(warpIdInBlock == 0 && warp.thread_rank() == 3){
            //     printf("warpNumRuns %d\n", warpNumRuns);
            // }

            // if(warpIdInBlock == 0 && warp.thread_rank() == 3){
            //     printf("tailFlags PS\n");
            //     #pragma unroll
            //     for(int i = 0; i < items_per_thread; i++){
            //         printf("%d ", prefixsum[i]);
            //     }
            //     printf("\n");
            // }

            //inclusive scan to compute run lengths
            int run_lengths[items_per_thread];
            {
                RunIdCountPair zippedprefixsum[items_per_thread];
                RunIdCountPair zipped[items_per_thread];
                #pragma unroll
                for(int i = 0; i < items_per_thread; i++){
                    zipped[i].count = 1;
                    zipped[i].runId = prefixsum[i];
                }
                if(chunkId > 0 && warp.thread_rank() == 0){
                    if(lastElementOfPreviousChunk == items[0]){
                        zipped[0].count = 1 + lastCountOfPreviousChunk;
                    }
                }
                RunIdCountPairScanOp pairScanOp{};

                //inclusive scan
                zippedprefixsum[0] = zipped[0];
                #pragma unroll
                for(int i = 1; i < items_per_thread; i++){
                    zippedprefixsum[i] = pairScanOp(zippedprefixsum[i-1], zipped[i]);
                }
                RunIdCountPair inclTotal = zippedprefixsum[items_per_thread-1];
                RunIdCountPair predTotal;
                WarpScanPair(temp.temp_warpScanPair[warpIdInBlock]).ExclusiveScan(inclTotal, predTotal, RunIdCountPair{0, 0}, pairScanOp);
                for(int i = 0; i < items_per_thread; i++){
                    zippedprefixsum[i] = pairScanOp(predTotal, zippedprefixsum[i]);
                }

                // if(warpIdInBlock == 0 && warp.thread_rank() == 3){
                //     printf("zippedprefixsum count\n");
                //     #pragma unroll
                //     for(int i = 0; i < items_per_thread; i++){
                //         printf("%d ", zippedprefixsum[i].count);
                //     }
                //     printf("\n");
                // }
                // if(warpIdInBlock == 0 && warp.thread_rank() == 3){
                //     printf("zippedprefixsum runId\n");
                //     #pragma unroll
                //     for(int i = 0; i < items_per_thread; i++){
                //         printf("%d ", zippedprefixsum[i].runId);
                //     }
                //     printf("\n");
                // }

                lastElementOfPreviousChunk = warp.shfl(items[items_per_thread-1], warp.size() - 1);
                lastCountOfPreviousChunk = warp.shfl(zippedprefixsum[items_per_thread-1].count, warp.size() - 1);

                #pragma unroll
                for(int i = 0; i < items_per_thread; i++){
                    run_lengths[i] = tailFlags[i] == 1 ? zippedprefixsum[i].count : 0;
                }
            }

            // if(warpIdInBlock == 0 && warp.thread_rank() == 3){
            //     printf("run_lengths\n");
            //     #pragma unroll
            //     for(int i = 0; i < items_per_thread; i++){
            //         printf("%d ", run_lengths[i]);
            //     }
            //     printf("\n");
            // }


            //exclusive scan of run lengths
            int run_lengths_ps[items_per_thread];
            int warpNumDecodedElements = 0;            
            {
                run_lengths_ps[0] = 0;
                #pragma unroll
                for(int i = 1; i < items_per_thread; i++){
                    run_lengths_ps[i] = run_lengths_ps[i-1] + run_lengths[i-1];
                }
                int inclTotal = run_lengths_ps[items_per_thread-1] + run_lengths[items_per_thread-1];
                int predTotal = 0;
                WarpScanInt(temp.temp_warpScanInt[warpIdInBlock]).ExclusiveSum(inclTotal, predTotal, warpNumDecodedElements);
                for(int i = 0; i < items_per_thread; i++){
                    run_lengths_ps[i] += predTotal;
                }
            }

            // if(warpIdInBlock == 0 && warp.thread_rank() == 3){
            //     printf("run_lengths excl PS\n");
            //     #pragma unroll
            //     for(int i = 0; i < items_per_thread; i++){
            //         printf("%d ", run_lengths_ps[i]);
            //     }
            //     printf("\n");
            // }


            // if(warpIdInBlock == 0 && warp.thread_rank() == 3){
            //     printf("warpNumDecodedElements %d\n", warpNumDecodedElements);
            // }

            // if(warpIdInBlock == 0){
            //     if(segmentId == 0){

            //         warp.sync();
            //         if(warp.thread_rank() == 0){
            //             printf("warpNumDecodedElements %d\n", warpNumDecodedElements);
            //         }
            //         warp.sync();
            //         for(int t = 0; t < warp.size(); t++){
            //             if(t == warp.thread_rank()){
            //                 for(int i = 0; i < items_per_thread; i++){
            //                     printf("thread %d, letter,tailFlag,prefixsum,run_lengths %d, %d, %d, %d, %d\n", 
            //                         warp.thread_rank(), int(items[i]), tailFlags[i], prefixsum[i], run_lengths[i], run_lengths_ps[i]);
            //                 }
            //             }
            //         }
            //         warp.sync();
            //     }
            // }


            //write masked RLE result to smem
            for(int i = 0; i < items_per_thread; i++){ 
                if(tailFlags[i] == 1){
                    char letter = items[i];
                    const int count = run_lengths[i];
                    if(count >= maskingThreshold){
                        letter = maskingLetter;
                    }
                    const int smemOutputOffset = prefixsum[i];
                    const int runOffset = run_lengths_ps[i];

                    temp.runOffsets[warpIdInBlock][smemOutputOffset] = runOffset; 
                    temp.runValues[warpIdInBlock][smemOutputOffset] = letter;
                    // printf("thread %d, write (%c %d) to smem.\n", warp.thread_rank(), letter, runOffset);
                }
            }
            warp.sync(); //wait for RLE results

            // if(warpIdInBlock == 0 && warp.thread_rank() == 0){
            //     if(segmentId == 0){
            //         printf("runs %d. warpNumDecodedElements %d. (value, offset)\n", warpNumRuns, warpNumDecodedElements);
            //         for(int i = 0; i < warpNumRuns; i++){
            //             printf("(%d %d), ", int(temp.runValues[warpIdInBlock][i]), temp.runOffsets[warpIdInBlock][i]);
            //         }
            //         printf("\n");
            //     }
            // }
            

            //decode RLE results into smem and write back to global output
            {
                const int* const smemOffsetsBegin = &temp.runOffsets[warpIdInBlock][0];
                const int* const smemOffsetsEnd = smemOffsetsBegin + warpNumRuns;
                const int iterations = cuda::ceil_div(warpNumDecodedElements, items_per_warp);
                for(int iteration = 0; iteration < iterations; iteration++){
                    const int iterationBeginIndex = iteration * items_per_warp;
                    const int numInIteration = cuda::std::min(warpNumDecodedElements - iterationBeginIndex, items_per_warp);

                    for(int i = warp.thread_rank(); i < numInIteration; i += warp.size()){ 
                        const int outputIndex = iterationBeginIndex + i;
                        if(outputIndex < warpNumDecodedElements){
                            auto it = thrust::lower_bound(thrust::seq, smemOffsetsBegin, smemOffsetsEnd, outputIndex+1);
                            const int runId = cuda::std::distance(smemOffsetsBegin, it) - 1;
                            // if(segmentId == 42 && iteration > 0){
                            //     printf("outputIndex %d, runId %d, iterationBeginIndex %d\n", outputIndex, runId, iterationBeginIndex);
                            // }
                            temp.u.masked_letters[warpIdInBlock][i] = temp.runValues[warpIdInBlock][runId];
                        }
                    }
                    warp.sync();

                    // if(segmentId == 0){
                    //     if(warp.thread_rank() == 0){
                    //         printf("smem masked letters. numInIteration %d.\n", numInIteration);
                    //         for(int i = 0; i < numInIteration; i++){
                    //             printf("%d ", int(temp.u.masked_letters[warpIdInBlock][i]));
                    //         }
                    //         printf("\n");
                    //     }
                    // }

                    // write smem segment to global output;
                    for(int i = warp.thread_rank(); i < numInIteration; i += warp.size()){
                        segmentOutputData[writtenOutputElements + i] = temp.u.masked_letters[warpIdInBlock][i];
                    }
                    warp.sync();
                    writtenOutputElements += numInIteration;
                }
            }

        }

        // if(warpIdInBlock == 0 && warp.thread_rank() == 3){
        //     printf("segmentSize %d, writtenOutputElements %d\n", segmentSize, writtenOutputElements);
        // }
        // if(segmentSize != writtenOutputElements && segmentId == 13322){
        //     if(warp.thread_rank() == 0){
        //         printf("segmentId %d, segmentSize %d, writtenOutputElements %d\n", segmentId, segmentSize, writtenOutputElements);
        //         for(int i = 0; i < segmentSize; i++){
        //             printf("%d ", segmentInputData[i]);
        //         }
        //         printf("\n");
        //     }
        // }
        assert(segmentSize == writtenOutputElements);

        //apply 4-byte padding
        const int paddedLength = cuda::ceil_div(segmentSize, 4) * 4;
        if(segmentSize + warp.thread_rank() < paddedLength){
            segmentOutputData[segmentSize + warp.thread_rank()] = paddingLetter;
        }
    }
   
}

/*
    Requirements:

*/
template<class InputDataIterator, class InputOffsetIterator>
void maskSequences_custom_kernel_impl4(
    char* d_outputData,
    InputDataIterator d_inputData, //returns the letter (char) to consider for masking. each letter must be in range [0, 127]
    const int* d_inputLengths, //size numInputSequences
    InputOffsetIterator d_inputOffsets, //size numInputSequences
    size_t totalInputElements,
    int numInputSequences,
    int maskingThreshold, // replace letters in all repeats >= maskingThreshold by masking maskingLetter
    char maskingLetter,
    char paddingLetter, //to pad sequences to multiple of 4 bytes
    cudaStream_t stream
){


    int deviceId;
    int numSMs;
    int maxBlocksPerSM = 0;
    constexpr int blocksize = 256;
    constexpr int items_per_thread = 4;
    auto kernel = maskingKernel4_smem_rle_rld_1warpPerSegment<blocksize, items_per_thread, InputDataIterator, InputOffsetIterator>;

    CUDACHECK(cudaGetDevice(&deviceId));
    CUDACHECK(cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, deviceId));
    CUDACHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &maxBlocksPerSM,
        kernel,
        blocksize, 
        0
    ));

    const int requiredNumBlocks = cuda::ceil_div(numInputSequences, blocksize / 32);
    const int numBlocks = std::min(requiredNumBlocks, numSMs*maxBlocksPerSM);

    // std::cout << "numBlocks " << numBlocks << "\n";

    kernel<<<numBlocks, blocksize, 0, stream>>>(
        d_outputData,
        d_inputData,
        d_inputLengths,
        d_inputOffsets,
        totalInputElements,
        numInputSequences,
        maskingThreshold,
        maskingLetter,
        paddingLetter
    );
    CUDACHECKASYNC;
}






template<int blocksize, int maximumChunkSize, class InputTransformOp, class InputDataIterator, class InputOffsetIterator, class OutputOffsetIterator>
__global__
void maskingKernel5_smem_leftright_1warpPerSegment(
    char* d_outputData,
    OutputOffsetIterator d_outputOffsets,
    InputDataIterator d_inputData,
    const int* d_lengths,
    InputOffsetIterator d_offsets,
    int64_t totalInputElements,
    int numSegments,
    int maskingThreshold,
    char maskingLetter,
    char paddingLetter,
    InputTransformOp inputTransformOp
){
    __builtin_assume(blocksize == blockDim.x);

    constexpr int smemElements = maximumChunkSize;
    static_assert(smemElements % 4 == 0);
    static_assert(smemElements % 16 == 0);

    auto warp = cooperative_groups::tiled_partition<32>(cooperative_groups::this_thread_block());
    constexpr int numWarpsInBlock = blocksize / 32;
    const int numWarpsInGrid = numWarpsInBlock * gridDim.x;
    const int warpIdInBlock = warp.meta_group_rank();
    const int warpIdInGrid = (threadIdx.x + blockIdx.x * blockDim.x) / 32;

    constexpr bool isIdentityTransform = cuda::std::is_same_v<InputTransformOp, cuda::std::identity>;


    __shared__ alignas(4) struct Smem{
        char chunkElements[numWarpsInBlock][smemElements];
    } smem;


    for(int segmentId = warpIdInGrid; segmentId < numSegments; segmentId += numWarpsInGrid){
        const size_t segmentOffset = d_offsets[segmentId];
        const int segmentSize = d_lengths[segmentId];
        const auto segmentInputData = d_inputData + segmentOffset;
        char* const segmentOutputData = d_outputData + segmentOffset;
        

        //chunks are overlapping by maskingThreshold-1 positions
        // const int numChunks = cuda::ceil_div(segmentSize, (smemElements - maskingThreshold - 1));

        // for(int chunkId = 0; chunkId < numChunks; chunkId++){
        //     const int chunkBegin = chunkId * (smemElements - maskingThreshold - 1);
        //     const int chunkEnd_excl = cuda::std::min(segmentSize, chunkBegin + smemElements);
        //     const int numInChunk = chunkEnd_excl - chunkBegin;

        for(int chunkBegin = 0; chunkBegin < segmentSize; chunkBegin += smemElements - 2*(maskingThreshold - 1)){
            const int chunkEnd_excl = cuda::std::min(segmentSize, chunkBegin + smemElements);
            const int numInChunk = chunkEnd_excl - chunkBegin;

            //load chunk to shared memory
            warp.sync(); //wait before re-using smem
            if (isIdentityTransform && cuda::std::is_pointer_v<InputDataIterator> && chunkBegin % 4 == 0){
                assert(uintptr_t(segmentInputData) % 4 == 0);
                cooperative_groups::memcpy_async(
                    warp, 
                    &smem.chunkElements[warpIdInBlock][0], 
                    cuda::aligned_size_t<4>(smemElements), 
                    segmentInputData + chunkBegin, 
                    cuda::aligned_size_t<4>(cuda::ceil_div(numInChunk,4)*4) //segments are padded to 4 bytes so reading more is safe
                );
                cooperative_groups::wait(warp);
            }else{
                for(int i = chunkBegin + warp.thread_rank(); i < chunkEnd_excl; i += warp.size()){
                    smem.chunkElements[warpIdInBlock][i - chunkBegin] = inputTransformOp(segmentInputData[i]);
                }
                warp.sync();
            }


            if constexpr(!isIdentityTransform){
                // uint4* vectorsmem = reinterpret_cast<uint4*>(&smem.chunkElements[warpIdInBlock][0]);
                // for(int i = warp.thread_rank(); i < cuda::ceil_div(numInChunk,16); i += warp.size()){
                //     uint4 tmp = vectorsmem[i];
                //     alignas(16) char letters[16];
                //     memcpy(&letters[0], &tmp, 16);

                //     #pragma unroll
                //     for(int k = 0; k < 16; k++){
                //         letters[k] = inputTransformOp(letters[k]);
                //     }

                //     memcpy(&tmp, &letters[0], 16);

                //     vectorsmem[i] = tmp;
                // }
                // warp.sync();
            }
            




            int firstOutputElementIndex = chunkBegin;
            if(firstOutputElementIndex > 0){
                firstOutputElementIndex += (maskingThreshold-1);
            }
            int lastOutputElementIndex_excl = chunkEnd_excl;
            if(lastOutputElementIndex_excl < segmentSize){
                lastOutputElementIndex_excl -= (maskingThreshold-1);
            }
            // if(segmentId == 11){
            // if(warp.thread_rank() == 0){
            //     printf("chunkBegin %d, chunkEnd_excl %d, firstOutputElementIndex %d, lastOutputElementIndex_excl %d\n", 
            //         chunkBegin, chunkEnd_excl, firstOutputElementIndex, lastOutputElementIndex_excl);
            // }
            // }

            for(int elementIndex = firstOutputElementIndex + warp.thread_rank(); elementIndex < lastOutputElementIndex_excl; elementIndex += warp.size()){
                const int indexInSmem = elementIndex - chunkBegin;
                //const char myLetter = inputTransformOp(smem.chunkElements[warpIdInBlock][indexInSmem]);
                const char myLetter = (smem.chunkElements[warpIdInBlock][indexInSmem]);

                int count = 1; //account for my letter
                //check to the left
                for(int i = indexInSmem - 1; i >= cuda::std::max(0, indexInSmem - maskingThreshold + 1); i--){
                    //const char otherLetter = inputTransformOp(smem.chunkElements[warpIdInBlock][i]);
                    const char otherLetter = (smem.chunkElements[warpIdInBlock][i]);
                    if(myLetter == otherLetter){
                        count++;
                    }else{
                        break;
                    }
                }

                //check to the right
                for(int64_t i = indexInSmem + 1; i < cuda::std::min(numInChunk, indexInSmem + maskingThreshold); i++){
                    //const char otherLetter = inputTransformOp(smem.chunkElements[warpIdInBlock][i]);
                    const char otherLetter = (smem.chunkElements[warpIdInBlock][i]);
                    if(myLetter == otherLetter){
                        count++;
                        if(count >= maskingThreshold){
                            break;
                        }
                    }else{
                        break;
                    }
                }

                // if(segmentId == 11){
                //     if(elementIndex >= 10630){
                //         printf("elementIndex %d, letter %d, count %d\n", elementIndex, int(myLetter), count);
                //     }
                // }

                if(count >= maskingThreshold){
                    segmentOutputData[elementIndex] = maskingLetter;
                }else{
                    segmentOutputData[elementIndex] = myLetter;
                }
            }
        }

        //apply 4-byte padding
        const int paddedLength = cuda::ceil_div(segmentSize, 4) * 4;
        if(segmentSize + warp.thread_rank() < paddedLength){
            segmentOutputData[segmentSize + warp.thread_rank()] = paddingLetter;
        }
    }
   
}



/*
    Requirements:
        sequence[i] has length inputLengths[i] and begins at input position inputOffsets[i]
        sequence[i] is padded to 4 byte boundary
        d_outputData must not overlap d_inputData
        d_outputData must be able to contain 4-byte padded sequences
*/
template<class InputTransformOp, class InputDataIterator, class InputOffsetIterator, class OutputOffsetIterator>
void maskSequences_custom_kernel_impl5(
    char* d_outputData,
    OutputOffsetIterator d_outputOffsets,
    InputDataIterator d_inputData, 
    const int* d_inputLengths, //size numInputSequences
    InputOffsetIterator d_inputOffsets, //size numInputSequences
    size_t totalInputElements,
    int numInputSequences,
    int maskingThreshold, // replace letters in all repeats >= maskingThreshold by masking maskingLetter
    char maskingLetter,
    char paddingLetter, //to pad output sequences to 4 byte boundary
    InputTransformOp inputTransformOp, // input letters are given by inputTransformOp(d_inputData[i])
    cudaStream_t stream
){
    constexpr int maximumChunkSize = 1024;

    if(2*maskingThreshold >= maximumChunkSize){
        throw std::runtime_error("maskingThreshold too large for hardcoded maximumChunkSize\n");
    }

    int deviceId;
    int numSMs;
    int maxBlocksPerSM = 0;
    constexpr int blocksize = 256;
    auto kernel = maskingKernel5_smem_leftright_1warpPerSegment<blocksize, maximumChunkSize, 
        InputTransformOp, InputDataIterator, InputOffsetIterator, OutputOffsetIterator>;

    CUDACHECK(cudaGetDevice(&deviceId));
    CUDACHECK(cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, deviceId));
    CUDACHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &maxBlocksPerSM,
        kernel,
        blocksize, 
        0
    ));

    const int requiredNumBlocks = cuda::ceil_div(numInputSequences, blocksize / 32);
    const int numBlocks = std::min(requiredNumBlocks, numSMs*maxBlocksPerSM);

    // std::cout << "numBlocks " << numBlocks << "\n";

    kernel<<<numBlocks, blocksize, 0, stream>>>(
        d_outputData,
        d_outputOffsets,
        d_inputData,
        d_inputLengths,
        d_inputOffsets,
        totalInputElements,
        numInputSequences,
        maskingThreshold,
        maskingLetter,
        paddingLetter,
        inputTransformOp
    );
    CUDACHECKASYNC;
}






template<int items_per_thread, int maxMaskingThreshold, class InputDataIterator, class InputOffsetIterator>
__global__
void maskingKernel6_reg_leftright_perSegment(
    char* d_outputData,
    InputDataIterator d_inputData,
    const int* d_lengths,
    InputOffsetIterator d_offsets,
    int64_t totalInputElements,
    int numSegments,
    int maskingThreshold,
    char maskingLetter,
    char paddingLetter
){
    static_assert(maxMaskingThreshold <= 32); //this ensures that at most two item indices need to be accessed for each search iteration

    constexpr int items_per_warp = 32 * items_per_thread;

    auto warp = cooperative_groups::tiled_partition<32>(cooperative_groups::this_thread_block());
    const int numWarpsInBlock = blockDim.x / 32;
    const int numWarpsInGrid = numWarpsInBlock * gridDim.x;
    // const int warpIdInBlock = warp.meta_group_rank();
    const int warpIdInGrid = (threadIdx.x + blockIdx.x * blockDim.x) / 32;

    constexpr int oob_element = cuda::std::numeric_limits<int>::max(); //use masking letter instead?


    for(int segmentId = warpIdInGrid; segmentId < numSegments; segmentId += numWarpsInGrid){
        const size_t segmentOffset = d_offsets[segmentId];
        const int segmentSize = d_lengths[segmentId];
        const auto segmentInputData = d_inputData + segmentOffset;
        char* const segmentOutputData = d_outputData + segmentOffset;

        for(int chunkBegin = 0; chunkBegin < segmentSize; chunkBegin += items_per_warp - 2*(maskingThreshold - 1)){
            const int chunkEnd_excl = cuda::std::min(segmentSize, chunkBegin + items_per_warp);
            // const int numInChunk = chunkEnd_excl - chunkBegin;

            //load chunk to registers

            int items[items_per_thread];
            
            #pragma unroll
            for(int i = 0; i < items_per_thread; i++){
                //striped index
                const int index = chunkBegin + i * warp.size() + warp.thread_rank();
                if(index < chunkEnd_excl){
                    items[i] = segmentInputData[index];
                }else{
                    items[i] = oob_element;
                }
            }

            int firstOutputElementIndex = chunkBegin;
            if(firstOutputElementIndex > 0){
                firstOutputElementIndex += (maskingThreshold-1);
            }
            int lastOutputElementIndex_excl = chunkEnd_excl;
            if(lastOutputElementIndex_excl < segmentSize){
                lastOutputElementIndex_excl -= (maskingThreshold-1);
            }

            //scan surrounding elements to check if run lenght meets threshold, then write to output
            #pragma unroll
            for(int currentIndex = 0; currentIndex < items_per_thread; currentIndex++){
                //striped index
                const int outputindex = chunkBegin + currentIndex * warp.size() + warp.thread_rank();

                //check left and right of currentIndex to count occurences
                int myLetter = items[currentIndex];
                int count = 1;
                
                //check left
                int leftNeighborTid = warp.thread_rank() == 0 ? warp.size() - 1 : warp.thread_rank() - 1;
                bool stopLeft = false;
                #pragma unroll
                for(int distance = 1; distance < maxMaskingThreshold; distance++){
                    if(distance < maskingThreshold){
                        const int neighborLetterFromPrevIndex = currentIndex == 0 ? oob_element : warp.shfl(items[currentIndex-1], leftNeighborTid);
                        const int neighborLetterFromSameIndex = warp.shfl(items[currentIndex], leftNeighborTid);
                        const int neighborLetter = leftNeighborTid >= warp.thread_rank() ? neighborLetterFromPrevIndex : neighborLetterFromSameIndex;
                        // if(segmentId == 1029){
                        //     if(outputindex == 55){
                        //         printf("left. chunkBegin %d, my thread id %d, currentIndex %d, distance %d, myLetter %d, neighborLetter %d, leftNeighborTid %d\n",
                        //         chunkBegin, warp.thread_rank(), currentIndex, distance, myLetter, neighborLetter, leftNeighborTid);
                        //     }
                        // }
                        if(neighborLetter == myLetter){
                            if(!stopLeft){
                                count++;
                            }
                        }else{
                            //run which includes myLetter is finished
                            stopLeft = true; //thread is done, but we cannot break out of the loop because shfl needs to be executed by full warp
                        }
                        leftNeighborTid = leftNeighborTid == 0 ? warp.size() - 1 : leftNeighborTid - 1;
                    }
                }

                //check right
                int rightNeighborTid = (warp.thread_rank() + 1) % warp.size();
                bool stopRight = false;
                for(int distance = 1; distance < maxMaskingThreshold; distance++){
                    if(distance < maskingThreshold){
                        const int neighborLetterFromNextIndex = currentIndex == items_per_thread-1 ? oob_element : warp.shfl(items[currentIndex+1], rightNeighborTid);
                        const int neighborLetterFromSameIndex = warp.shfl(items[currentIndex], rightNeighborTid);
                        const int neighborLetter = rightNeighborTid <= warp.thread_rank() ? neighborLetterFromNextIndex : neighborLetterFromSameIndex;
                        // if(segmentId == 1029){
                        //     if(outputindex == 55){
                        //         printf("right. chunkBegin %d, my thread id %d, currentIndex %d, distance %d, myLetter %d, neighborLetter %d, rightNeighborTid %d\n",
                        //         chunkBegin, warp.thread_rank(), currentIndex, distance, myLetter, neighborLetter, rightNeighborTid);
                        //     }
                        // }
                        if(neighborLetter == myLetter){
                            if(!stopRight){
                                count++;
                            }
                        }else{
                            //run which includes myLetter is finished
                            stopRight = true; //thread is done, but we cannot break out of the loop because shfl needs to be executed by full warp
                        }
                        rightNeighborTid = (rightNeighborTid + 1) % warp.size();
                    }
                }


                if(firstOutputElementIndex <= outputindex && outputindex < lastOutputElementIndex_excl){
                    if(count >= maskingThreshold){
                        segmentOutputData[outputindex] = maskingLetter;
                    }else{
                        segmentOutputData[outputindex] = myLetter;
                    }
                }

            }
        }

        //apply 4-byte padding
        const int paddedLength = cuda::ceil_div(segmentSize, 4) * 4;
        if(segmentSize + warp.thread_rank() < paddedLength){
            segmentOutputData[segmentSize + warp.thread_rank()] = paddingLetter;
        }
    }
   
}

/*
    Requirements:

*/
template<int maxMaskingThreshold, class InputDataIterator, class InputOffsetIterator>
void maskSequences_custom_kernel_impl6(
    char* d_outputData,
    InputDataIterator d_inputData, //returns the letter (char) to consider for masking. each letter must be in range [0, 127]
    const int* d_inputLengths, //size numInputSequences
    InputOffsetIterator d_inputOffsets, //size numInputSequences
    size_t totalInputElements,
    int numInputSequences,
    int maskingThreshold, // replace letters in all repeats >= maskingThreshold by masking maskingLetter
    char maskingLetter,
    char paddingLetter, //to pad sequences to multiple of 4 bytes
    cudaStream_t stream
){
    if(maskingThreshold > maxMaskingThreshold){
        throw std::runtime_error("maskingThreshold too large for hardcoded maxMaskingThreshold\n");
    }

    constexpr int items_per_thread = 4;

    int deviceId;
    int numSMs;
    int maxBlocksPerSM = 0;
    constexpr int blocksize = 256;
    auto kernel = maskingKernel6_reg_leftright_perSegment<items_per_thread, maxMaskingThreshold, InputDataIterator, InputOffsetIterator>;

    CUDACHECK(cudaGetDevice(&deviceId));
    CUDACHECK(cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, deviceId));
    CUDACHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &maxBlocksPerSM,
        kernel,
        blocksize, 
        0
    ));

    const int requiredNumBlocks = cuda::ceil_div(numInputSequences, blocksize / 32);
    const int numBlocks = std::min(requiredNumBlocks, numSMs*maxBlocksPerSM);

    // std::cout << "numBlocks " << numBlocks << "\n";

    kernel<<<numBlocks, blocksize, 0, stream>>>(
        d_outputData,
        d_inputData,
        d_inputLengths,
        d_inputOffsets,
        totalInputElements,
        numInputSequences,
        maskingThreshold,
        maskingLetter,
        paddingLetter
    );
    CUDACHECKASYNC;
}






/*
    In constrast to maskingKernel5 which splits input sequences into chunks, maskingKernel7 splits output sequences into chunks
*/
template<int blocksize, int outputChunkSize, int maxMaskingThreshold, class InputTransformOp, class InputDataIterator, class InputOffsetIterator, class OutputOffsetIterator>
__global__
void maskingKernel7_smem_leftright_1warpPerSegment(
    char* d_outputData,
    OutputOffsetIterator d_outputOffsets,
    InputDataIterator d_inputData,
    const int* d_lengths,
    InputOffsetIterator d_offsets,
    int64_t totalInputElements,
    int numSegments,
    int maskingThreshold,
    char maskingLetter,
    char paddingLetter,
    InputTransformOp inputTransformOp
){
    __builtin_assume(blocksize == blockDim.x);
    __builtin_assume(maskingThreshold <= maxMaskingThreshold);

    static_assert(outputChunkSize % 16 == 0);
    constexpr int smemNumInputElements = outputChunkSize + 2 * (cuda::ceil_div((maxMaskingThreshold-1), 16) * 16);
    // constexpr int smemNumOutputElements = outputChunkSize;

    auto warp = cooperative_groups::tiled_partition<32>(cooperative_groups::this_thread_block());
    constexpr int numWarpsInBlock = blocksize / 32;
    const int numWarpsInGrid = numWarpsInBlock * gridDim.x;
    const int warpIdInBlock = warp.meta_group_rank();
    const int warpIdInGrid = (threadIdx.x + blockIdx.x * blockDim.x) / 32;

    constexpr bool isIdentityTransform = cuda::std::is_same_v<InputTransformOp, cuda::std::identity>;

    constexpr int numStages = 2;
    
    __shared__ alignas(16) struct Smem{
        char inputElements[numWarpsInBlock][numStages][smemNumInputElements];
        //char outputElements[numWarpsInBlock][smemNumOutputElements];
    } smem;


    for(int segmentId = warpIdInGrid; segmentId < numSegments; segmentId += numWarpsInGrid){

        const size_t segmentOffset = d_offsets[segmentId];
        const int segmentSize = d_lengths[segmentId];
        const auto segmentInputData = d_inputData + segmentOffset;
        char* const segmentOutputData = d_outputData + segmentOffset;
        const int numOutputChunks = cuda::ceil_div(segmentSize, outputChunkSize);

        for(int outputChunk = 0, fetchChunk = 0; outputChunk < numOutputChunks; outputChunk++){
            for (; fetchChunk < numOutputChunks && fetchChunk < (outputChunk + numStages); ++fetchChunk) {
                const int outputChunkBegin = fetchChunk * outputChunkSize;
                const int outputChunkEnd_excl = cuda::std::min(segmentSize, outputChunkBegin + outputChunkSize);
                // const int numInOutputChunk = outputChunkEnd_excl - outputChunkBegin;
    
                const int inputChunkBegin = cuda::std::max(0, outputChunkBegin - (maskingThreshold-1));
                const int inputChunkEnd_excl = cuda::std::min(segmentSize, outputChunkEnd_excl + (maskingThreshold-1));
    
                //align load begin to 4 bytes. valid since each segment begin is aligned to 4 bytes.
                const int inputLoadFrontPadding = inputChunkBegin % 4;
                const int inputChunkLoadBegin = inputChunkBegin - inputLoadFrontPadding;
                //align load end to 4 bytes. valid since all segments are padded to 4 bytes
                const int inputChunkLoadEnd_excl = cuda::ceil_div(inputChunkEnd_excl, 4) * 4;
                const int loadSize = inputChunkLoadEnd_excl - inputChunkLoadBegin;
                // assert(loadSize <= smemNumInputElements);

                // if(segmentId == 0){
                //     if(warp.thread_rank() == 0){
                //         printf("fetchChunk %d, load %d elements, %d - %d, smemNumInputElements %d\n", 
                //                 fetchChunk,loadSize, inputChunkLoadBegin,inputChunkLoadEnd_excl, smemNumInputElements);
                //         for(int i = 0; i < 16; i++){
                //             printf("%d ", int((segmentInputData + inputChunkLoadBegin)[i]));
                //         }
                //         printf("\n");
                //     }
                // }
                cooperative_groups::memcpy_async(
                    warp, 
                    &smem.inputElements[warpIdInBlock][fetchChunk % numStages][0], 
                    cuda::aligned_size_t<4>(smemNumInputElements), 
                    segmentInputData + inputChunkLoadBegin, 
                    cuda::aligned_size_t<4>(loadSize)
                );
            }
            const int outputChunkBegin = outputChunk * outputChunkSize;
            const int outputChunkEnd_excl = cuda::std::min(segmentSize, outputChunkBegin + outputChunkSize);
            const int numInOutputChunk = outputChunkEnd_excl - outputChunkBegin;

            const int inputChunkBegin = cuda::std::max(0, outputChunkBegin - (maskingThreshold-1));
            const int inputChunkEnd_excl = cuda::std::min(segmentSize, outputChunkEnd_excl + (maskingThreshold-1));

            //align load begin to 4 bytes. valid since each segment begin is aligned to 4 bytes.
            const int inputLoadFrontPadding = inputChunkBegin % 4;
            const int inputChunkLoadBegin = inputChunkBegin - inputLoadFrontPadding;
            //align load end to 4 bytes. valid since all segments are padded to 4 bytes
            const int inputChunkLoadEnd_excl = cuda::ceil_div(inputChunkEnd_excl, 4) * 4;
            const int loadSize = inputChunkLoadEnd_excl - inputChunkLoadBegin;

            //wait for async load from gmem to smem of our input chunk
            if(outputChunk < numOutputChunks-1){
                cooperative_groups::wait_prior<1>(warp);
                // cooperative_groups::wait(warp);
            }else{
                cooperative_groups::wait(warp);
            }


            if constexpr(!isIdentityTransform){
                for(int i = 4 * warp.thread_rank(); i < loadSize; i += 4*warp.size()){
                    char4 four = *reinterpret_cast<const char4*>(&smem.inputElements[warpIdInBlock][outputChunk % numStages][i]);
                    four.x = inputTransformOp(four.x);
                    four.y = inputTransformOp(four.y);
                    four.z = inputTransformOp(four.z);
                    four.w = inputTransformOp(four.w);
                    *reinterpret_cast<char4*>(&smem.inputElements[warpIdInBlock][outputChunk % numStages][i]) = four;
                }
            }

            const int inputOutputSmemOffset = outputChunkBegin - inputChunkLoadBegin;
            const int inputSmemEndIndex_excl = inputChunkEnd_excl - inputChunkLoadBegin;
            for(int outputElementLocalIndex = warp.thread_rank(); outputElementLocalIndex < numInOutputChunk; outputElementLocalIndex += warp.size()){

                const int inputIndexInSmem = inputOutputSmemOffset + outputElementLocalIndex;
                const char myLetter = (smem.inputElements[warpIdInBlock][outputChunk % numStages][inputIndexInSmem]);

                // if(segmentId == 0){
                //     if(outputChunk == 1){
                //         if(warp.thread_rank() == 0){
                //             printf("outputElementLocalIndex %d, inputIndexInSmem %d, myLetter %d\n", 
                //                 outputElementLocalIndex, inputIndexInSmem, myLetter);
                //         }
                //     }
                // }

                int count = 1; //account for my letter
                //check to the left
                for(int i = inputIndexInSmem - 1; i >= cuda::std::max(0, inputIndexInSmem - maskingThreshold + 1); i--){
                    //const char otherLetter = inputTransformOp(smem.inputElements[warpIdInBlock][i]);
                    const char otherLetter = (smem.inputElements[warpIdInBlock][outputChunk % numStages][i]);
                    if(myLetter == otherLetter){
                        count++;
                    }else{
                        break;
                    }
                }

                //check to the right
                const int loopLimit = cuda::std::min(inputSmemEndIndex_excl, inputIndexInSmem + maskingThreshold);
                for(int64_t i = inputIndexInSmem + 1; i < loopLimit; i++){                    
                    //const char otherLetter = inputTransformOp(smem.chunkElements[warpIdInBlock][i]);
                    const char otherLetter = (smem.inputElements[warpIdInBlock][outputChunk % numStages][i]);
                    if(myLetter == otherLetter){
                        count++;
                        if(count >= maskingThreshold){
                            break;
                        }
                    }else{
                        break;
                    }
                }

                // if(warp.thread_rank() == 0){
                //     if(segmentId == 0){
                //         printf("write to gmem pos %d, count %d\n", outputChunkBegin + outputElementLocalIndex, count);
                //     }
                // }

                const char outputLetter = (count >= maskingThreshold) ? maskingLetter : myLetter;
                segmentOutputData[outputChunkBegin + outputElementLocalIndex] = outputLetter;
            }
            
            warp.sync(); //done with shared memory of stage, can be reused for prefetching
        }
        //apply 4-byte padding
        const int paddedLength = cuda::ceil_div(segmentSize, 4) * 4;
        if(segmentSize + warp.thread_rank() < paddedLength){
            segmentOutputData[segmentSize + warp.thread_rank()] = paddingLetter;
        }

    }
   
}



/*
    Requirements:
        sequence[i] has length inputLengths[i] and begins at input position inputOffsets[i]
        sequence[i] is padded to 4 byte boundary
        d_outputData must not overlap d_inputData
        d_outputData must be able to contain 4-byte padded sequences
*/
template<int outputChunkSize, int maxMaskingThreshold, class InputTransformOp, class InputDataIterator, class InputOffsetIterator, class OutputOffsetIterator>
void maskSequences_custom_kernel_impl7(
    char* d_outputData,
    OutputOffsetIterator d_outputOffsets,
    InputDataIterator d_inputData, 
    const int* d_inputLengths, //size numInputSequences
    InputOffsetIterator d_inputOffsets, //size numInputSequences
    size_t totalInputElements,
    int numInputSequences,
    int maskingThreshold, // replace letters in all repeats >= maskingThreshold by masking maskingLetter
    char maskingLetter,
    char paddingLetter, //to pad output sequences to 4 byte boundary
    InputTransformOp inputTransformOp, // input letters are given by inputTransformOp(d_inputData[i])
    cudaStream_t stream
){
    if(maskingThreshold > maxMaskingThreshold){
        throw std::runtime_error("maskingThreshold > maxMaskingThreshold");
    }
    int deviceId;
    int numSMs;
    int maxBlocksPerSM = 0;
    constexpr int blocksize = 256;
    auto kernel = maskingKernel7_smem_leftright_1warpPerSegment<blocksize, outputChunkSize, maxMaskingThreshold,  
        InputTransformOp, InputDataIterator, InputOffsetIterator, OutputOffsetIterator>;

    CUDACHECK(cudaGetDevice(&deviceId));
    CUDACHECK(cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, deviceId));
    CUDACHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &maxBlocksPerSM,
        kernel,
        blocksize, 
        0
    ));

    const int requiredNumBlocks = cuda::ceil_div(numInputSequences, blocksize / 32);
    const int numBlocks = std::min(requiredNumBlocks, numSMs*maxBlocksPerSM);

    // std::cout << "numBlocks " << numBlocks << "\n";

    kernel<<<numBlocks, blocksize, 0, stream>>>(
        d_outputData,
        d_outputOffsets,
        d_inputData,
        d_inputLengths,
        d_inputOffsets,
        totalInputElements,
        numInputSequences,
        maskingThreshold,
        maskingLetter,
        paddingLetter,
        inputTransformOp
    );
    CUDACHECKASYNC;
}













template<class InputTransformOp, class InputDataIterator, class InputOffsetIterator, class OutputOffsetIterator>
void maskSequences(
    char* d_outputData,
    OutputOffsetIterator d_outputOffsets,
    InputDataIterator d_inputData, //returns the letter (char) to consider for masking. each letter must be in range [0, 127]
    const int* d_inputLengths, //size numInputSequences
    InputOffsetIterator d_inputOffsets, //size numInputSequences
    size_t totalInputElements,
    int numInputSequences,
    int maskingThreshold, // replace letters in all repeats >= maskingThreshold by masking maskingLetter
    char maskingLetter,
    char paddingLetter,
    InputTransformOp inputTransformOp,
    cudaStream_t stream
){

    maskSequences_custom_kernel_impl5(
        d_outputData,
        d_outputOffsets,
        d_inputData,
        d_inputLengths,
        d_inputOffsets,
        totalInputElements,
        numInputSequences,
        maskingThreshold,
        maskingLetter,
        paddingLetter,
        inputTransformOp,
        stream
    );
}



LIBMARV_NAMESPACE_WITH_NESTING_END

#endif