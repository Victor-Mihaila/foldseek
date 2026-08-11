#ifndef LIBMARV_GPU_DATABASE_CUH
#define LIBMARV_GPU_DATABASE_CUH

#include <cuda/std/span>

#include "hpc_helpers/cuda_raiiwrappers.cuh"
#include "hpc_helpers/all_helpers.cuh"
#include "hpc_helpers/nvtx_markers.cuh"
#include "hpc_helpers/simple_allocation.cuh"

#include "config.hpp"
#include "dbdata.hpp"
#include "length_partitions.hpp"
#include "util.cuh"
// #include "types.hpp"
#include "dbbatching.cuh"
#include "cuda_errorcheck.cuh"
#include "offset_iterator.cuh"
#include "gpudatabaseallocation.cuh"

#include <thrust/binary_search.h>
// #include <thrust/distance.h>
#include <thrust/scan.h>
#include <thrust/gather.h>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>

#include <iostream>
// #include <string>
#include <vector>
#include <memory>
// #include <string_view>
// #include <optional>
// #include <sstream>

#include "namespace.hpp"
LIBMARV_NAMESPACE_WITH_NESTING_BEGIN



    template<class PartitionOffsets, class Indices>
    __global__
    void transformLocalSequenceIndicesToGlobalIndicesKernel(
        int gpu,
        int N,
        PartitionOffsets partitionOffsets,
        Indices indices
    ){
        const int tid = threadIdx.x + blockIdx.x * blockDim.x;

        if(tid < N){
            indices[tid] = partitionOffsets.getGlobalIndex(gpu, indices[tid]);
        }
    }

    struct HostGpuPartitionOffsets{
        int numGpus;
        int numLengthPartitions;
        std::vector<size_t> partitionSizes;
        std::vector<size_t> horizontalPS;
        std::vector<size_t> verticalPS;
        std::vector<size_t> totalPerLengthPartitionPS;

        HostGpuPartitionOffsets() = default;

        HostGpuPartitionOffsets(int numGpus_, int numLengthpartitions_, std::vector<size_t> partitionSizes_)
            : numGpus(numGpus_), 
            numLengthPartitions(numLengthpartitions_), 
            partitionSizes(std::move(partitionSizes_)),
            horizontalPS(numGpus * numLengthPartitions, 0),
            verticalPS(numGpus * numLengthPartitions, 0),
            totalPerLengthPartitionPS(numLengthPartitions, 0)
        {
            assert(partitionSizes.size() == numGpus * numLengthPartitions);

            for(int gpu = 0; gpu < numGpus; gpu++){
                for(int l = 1; l < numLengthPartitions; l++){
                    horizontalPS[gpu * numLengthPartitions + l] = horizontalPS[gpu * numLengthPartitions + l-1] + partitionSizes[gpu * numLengthPartitions + l-1];
                }
            }
            for(int l = 0; l < numLengthPartitions; l++){
                for(int gpu = 1; gpu < numGpus; gpu++){
                    verticalPS[gpu * numLengthPartitions + l] = verticalPS[(gpu-1) * numLengthPartitions + l] + partitionSizes[(gpu-1) * numLengthPartitions + l];
                }
            }
            for(int l = 1; l < numLengthPartitions; l++){
                totalPerLengthPartitionPS[l] = totalPerLengthPartitionPS[l-1] 
                    + (verticalPS[(numGpus - 1) * numLengthPartitions + (l-1)] + partitionSizes[(numGpus-1) * numLengthPartitions + (l-1)]);
            }
        }

        size_t getGlobalIndex(int gpu, size_t localIndex) const {
            const size_t* const myHorizontalPS = horizontalPS.data() + gpu * numLengthPartitions;
            const auto it = std::lower_bound(myHorizontalPS, myHorizontalPS + numLengthPartitions, localIndex+1);
            const int whichPartition = std::distance(myHorizontalPS, it) - 1;
            const size_t occurenceInPartition = localIndex - myHorizontalPS[whichPartition];
            const size_t globalPartitionBegin = totalPerLengthPartitionPS[whichPartition];
            const size_t elementsOfOtherPreviousGpusInPartition = verticalPS[gpu * numLengthPartitions + whichPartition];
            //std::cout << "whichPartition " << whichPartition << ", occurenceInPartition " << occurenceInPartition 
            //    << ", globalPartitionBegin " << globalPartitionBegin << ", elementsOfOtherPreviousGpusInPartition " << elementsOfOtherPreviousGpusInPartition << "\n";
            return globalPartitionBegin + elementsOfOtherPreviousGpusInPartition + occurenceInPartition;
        };

        void print(std::ostream& os){
            os << "numGpus " << numGpus << "\n";
            os << "numLengthPartitions " << numLengthPartitions << "\n";
            os << "partitionSizes\n";
            for(int gpu = 0; gpu < numGpus; gpu++){
                for(int l = 0; l < numLengthPartitions; l++){
                    os << partitionSizes[gpu * numLengthPartitions + l] << " ";
                }
                os << "\n";
            }
            os << "\n";
            os << "horizontalPS\n";
            for(int gpu = 0; gpu < numGpus; gpu++){
                for(int l = 0; l < numLengthPartitions; l++){
                    os << horizontalPS[gpu * numLengthPartitions + l] << " ";
                }
                os << "\n";
            }
            os << "\n";
            os << "verticalPS\n";
            for(int gpu = 0; gpu < numGpus; gpu++){
                for(int l = 0; l < numLengthPartitions; l++){
                    os << verticalPS[gpu * numLengthPartitions + l] << " ";
                }
                os << "\n";
            }
            os << "\n";
            os << "totalPerLengthPartitionPS\n";
            for(int l = 0; l < numLengthPartitions; l++){
                os << totalPerLengthPartitionPS[l] << " ";
            }
            os << "\n";
        }
    };

    struct DeviceGpuPartitionOffsets{
        template<class T>
        using MyDeviceBuffer = helpers::SimpleAllocationDevice<T, 0>;

        int numGpus;
        int numLengthPartitions;
        MyDeviceBuffer<size_t> partitionSizes;
        MyDeviceBuffer<size_t> horizontalPS;
        MyDeviceBuffer<size_t> verticalPS;
        MyDeviceBuffer<size_t> totalPerLengthPartitionPS;

        struct View{
            int numGpus;
            int numLengthPartitions;
            const size_t* partitionSizes;
            const size_t* horizontalPS;
            const size_t* verticalPS;
            const size_t* totalPerLengthPartitionPS;

            __device__
            size_t getGlobalIndex(int gpu, size_t localIndex) const {
                const size_t* const myHorizontalPS = horizontalPS + gpu * numLengthPartitions;
                const auto it = thrust::lower_bound(thrust::seq, myHorizontalPS, myHorizontalPS + numLengthPartitions, localIndex+1);
                const int whichPartition = cuda::std::distance(myHorizontalPS, it) - 1;
                const size_t occurenceInPartition = localIndex - myHorizontalPS[whichPartition];
                const size_t globalPartitionBegin = totalPerLengthPartitionPS[whichPartition];
                const size_t elementsOfOtherPreviousGpusInPartition = verticalPS[gpu * numLengthPartitions + whichPartition];
                return globalPartitionBegin + elementsOfOtherPreviousGpusInPartition + occurenceInPartition;
            };
        };

        DeviceGpuPartitionOffsets() = default;
        DeviceGpuPartitionOffsets(const HostGpuPartitionOffsets& hostData)
            : numGpus(hostData.numGpus),
            numLengthPartitions(hostData.numLengthPartitions),
            partitionSizes(numGpus * numLengthPartitions),
            horizontalPS(numGpus * numLengthPartitions),
            verticalPS(numGpus * numLengthPartitions),
            totalPerLengthPartitionPS(numLengthPartitions)
        {
            cudaMemcpyAsync(partitionSizes.data(), hostData.partitionSizes.data(), sizeof(size_t) * numGpus * numLengthPartitions, H2D, cudaStreamLegacy); CUERR;
            cudaMemcpyAsync(horizontalPS.data(), hostData.horizontalPS.data(), sizeof(size_t) * numGpus * numLengthPartitions, H2D, cudaStreamLegacy); CUERR;
            cudaMemcpyAsync(verticalPS.data(), hostData.verticalPS.data(), sizeof(size_t) * numGpus * numLengthPartitions, H2D, cudaStreamLegacy); CUERR;
            cudaMemcpyAsync(totalPerLengthPartitionPS.data(), hostData.totalPerLengthPartitionPS.data(), sizeof(size_t) * numLengthPartitions, H2D, cudaStreamLegacy); CUERR;
        }

        View getDeviceView() const{
            View view;
            view.numGpus = numGpus;
            view.numLengthPartitions = numLengthPartitions;
            view.partitionSizes = partitionSizes.data();
            view.horizontalPS = horizontalPS.data();
            view.verticalPS = verticalPS.data();
            view.totalPerLengthPartitionPS = totalPerLengthPartitionPS.data();
            return view;
        }
    };


    class GpuDatabase{
    public:
        template<class T>
        using MyPinnedBuffer = helpers::SimpleAllocationPinnedHost<T, 0>;
        template<class T>
        using MyDeviceBuffer = helpers::SimpleAllocationDevice<T, 0>;

        struct CallbackInputPerGpu{
            int deviceId; //device to which pointers and stream belong
            int sequenceLengthUpperBound; //elements of d_inputLengths are not greater than sequenceLengthUpperBound
            const char* d_inputChars; // size: numInputChars
            const SequenceLengthT* d_inputLengths; // size: numInputSequences. sequence i has length inputLengths[i]
            ReadOnlyCustomOffsetIterator<size_t*> d_inputOffsets; //size: (numInputSequences+1). sequence i begins at inputChars + (inputOffsets[i] - inputOffsets[0])
            size_t numInputSequences; // number of sequences in batch
            size_t numInputChars;
            cudaStream_t stream;
        };

        using ForEachCallback = std::function<void(cuda::std::span<CallbackInputPerGpu> callbackInput)>;
        using EpilogueFunction = std::function<void(GpuDatabase*, cuda::std::span<cudaStream_t> streams)>;

        struct GpuWorkingSet{


            GpuWorkingSet(
                size_t gpumemlimit,
                size_t maxBatchBytes,
                size_t maxBatchSequences,
                const std::vector<DBdataView>& dbPartitions,
                const std::vector<DeviceBatchCopyToPinnedPlan>& dbBatches,
                bool needsPinnedStagingBuffers
            ) 
            {
                CUDACHECK(cudaGetDevice(&deviceId));

                size_t numSubjects = 0;
                size_t numSubjectBytes = 0;
                for(const auto& p : dbPartitions){
                    numSubjects += p.numSequences();
                    numSubjectBytes += p.numChars();
                }

                size_t usedGpuMem = 0;

                if(usedGpuMem > gpumemlimit){
                    throw std::runtime_error("Out of memory working set");
                }
                
        
                //devAlignmentScoresFloat.resize(numSubjects);
        
                forkStreamEvent = CudaEvent{cudaEventDisableTiming}; CUERR;

                numCopyBuffers = 2;
        
                h_chardata_vec.resize(numCopyBuffers);
                h_lengthdata_vec.resize(numCopyBuffers);
                h_offsetdata_vec.resize(numCopyBuffers);
                d_chardata_vec.resize(numCopyBuffers);
                d_lengthdata_vec.resize(numCopyBuffers);
                d_offsetdata_vec.resize(numCopyBuffers);
                copyStreams.resize(numCopyBuffers);
                pinnedBufferEvents.resize(numCopyBuffers);
                deviceBufferEvents.resize(numCopyBuffers);


                size_t memoryRequiredForFullDB = 0;
                memoryRequiredForFullDB += numSubjectBytes; // d_fulldb_chardata
                memoryRequiredForFullDB += sizeof(SequenceLengthT) * numSubjects; //d_fulldb_lengthdata
                memoryRequiredForFullDB += sizeof(size_t) * (numSubjects+1); //d_fulldb_offsetdata
                //memoryRequiredForFullDB += sizeof(ReferenceIdT) * numSubjects * 2; //d_overflow_positions_vec
        
               

                if(usedGpuMem + memoryRequiredForFullDB <= gpumemlimit){
                    numBatchesInCachedDB = dbBatches.size();
                    charsOfBatches = numSubjectBytes;
                    subjectsOfBatches = numSubjects;
                    d_cacheddb = std::make_shared<GpuDatabaseAllocation>(numSubjectBytes, numSubjects);

                    for(int i = 0; i < numCopyBuffers; i++){
                        if(needsPinnedStagingBuffers){
                            h_chardata_vec[i].resize(maxBatchBytes);
                            h_lengthdata_vec[i].resize(maxBatchSequences);
                            h_offsetdata_vec[i].resize(maxBatchSequences+1);
                        }
                        pinnedBufferEvents[i] = CudaEvent{cudaEventDisableTiming}; CUERR;
                        deviceBufferEvents[i] = CudaEvent{cudaEventDisableTiming}; CUERR;
                        //d_overflow_positions_vec[i].resize(numSubjects);
                    }
                }else{
                    //allocate a double buffer for batch transfering
                    size_t memoryRequiredForBatchedProcessing = 0;
                    memoryRequiredForBatchedProcessing += maxBatchBytes * numCopyBuffers; // d_chardata_vec
                    memoryRequiredForBatchedProcessing += sizeof(SequenceLengthT) * maxBatchSequences * numCopyBuffers; //d_lengthdata_vec
                    memoryRequiredForBatchedProcessing += sizeof(size_t) * (maxBatchSequences+1) * numCopyBuffers; //d_offsetdata_vec
                    usedGpuMem += memoryRequiredForBatchedProcessing;
                    if(usedGpuMem > gpumemlimit){
                        throw std::runtime_error("Out of memory working set");
                    }
                    
                    for(int i = 0; i < numCopyBuffers; i++){
                        if(needsPinnedStagingBuffers){
                            h_chardata_vec[i].resize(maxBatchBytes);
                            h_lengthdata_vec[i].resize(maxBatchSequences);
                            h_offsetdata_vec[i].resize(maxBatchSequences+1);
                        }
                        d_chardata_vec[i].resize(maxBatchBytes);
                        d_lengthdata_vec[i].resize(maxBatchSequences);
                        d_offsetdata_vec[i].resize(maxBatchSequences+1);
                        pinnedBufferEvents[i] = CudaEvent{cudaEventDisableTiming}; CUERR;
                        deviceBufferEvents[i] = CudaEvent{cudaEventDisableTiming}; CUERR;
                        //d_overflow_positions_vec[i].resize(numSubjects);
                    }

                    //count how many batches fit into remaining gpu memory

                    numBatchesInCachedDB = 0;
                    charsOfBatches = 0;
                    subjectsOfBatches = 0;
                    size_t totalRequiredMemForBatches = sizeof(size_t);
                    for(; numBatchesInCachedDB < dbBatches.size(); numBatchesInCachedDB++){
                        const auto& batch = dbBatches[numBatchesInCachedDB];
                        const size_t requiredMemForBatch = batch.usedSeq * sizeof(SequenceLengthT) + batch.usedSeq * sizeof(size_t) + batch.usedBytes;
                        if(usedGpuMem + totalRequiredMemForBatches + requiredMemForBatch <= gpumemlimit){
                            //ok, fits
                            totalRequiredMemForBatches += requiredMemForBatch;
                            charsOfBatches += batch.usedBytes;
                            subjectsOfBatches += batch.usedSeq;
                        }else{
                            //does not fit
                            break;
                        }
                    }
                    assert(numBatchesInCachedDB < dbBatches.size());

                    //std::cout << "numBatchesInCachedDB " << numBatchesInCachedDB << ", charsOfBatches " << charsOfBatches << ", subjectsOfBatches " << subjectsOfBatches << "\n";

                    assert(usedGpuMem + totalRequiredMemForBatches <= gpumemlimit);
                    d_cacheddb = std::make_shared<GpuDatabaseAllocation>(charsOfBatches, subjectsOfBatches);
                }
            }

            GpuWorkingSet(
                size_t gpumemlimit,
                size_t maxBatchBytes,
                size_t maxBatchSequences,
                const std::vector<DBdataView>& dbPartitions,
                const std::vector<DeviceBatchCopyToPinnedPlan>& dbBatches,
                std::shared_ptr<GpuDatabaseAllocationBase> existingGpuDBAllocation,
                bool needsPinnedStagingBuffers
            )
            {
                CUDACHECK(cudaGetDevice(&deviceId));

                assert(existingGpuDBAllocation != nullptr);

                size_t numSubjects = 0;
                size_t numSubjectBytes = 0;
                for(const auto& p : dbPartitions){
                    numSubjects += p.numSequences();
                    numSubjectBytes += p.numChars();
                }

                size_t usedGpuMem = 0;

                if(usedGpuMem > gpumemlimit){
                    throw std::runtime_error("Out of memory working set");
                }
                
        
                //devAlignmentScoresFloat.resize(numSubjects);
        
                forkStreamEvent = CudaEvent{cudaEventDisableTiming}; CUERR;

                numCopyBuffers = 2;
        
                h_chardata_vec.resize(numCopyBuffers);
                h_lengthdata_vec.resize(numCopyBuffers);
                h_offsetdata_vec.resize(numCopyBuffers);
                d_chardata_vec.resize(numCopyBuffers);
                d_lengthdata_vec.resize(numCopyBuffers);
                d_offsetdata_vec.resize(numCopyBuffers);
                copyStreams.resize(numCopyBuffers);
                pinnedBufferEvents.resize(numCopyBuffers);
                deviceBufferEvents.resize(numCopyBuffers);
                //d_total_overflow_number.resize(1);
                //d_overflow_number.resize(numCopyBuffers);
                //h_overflow_number.resize(numCopyBuffers);
                //d_overflow_positions_vec.resize(numCopyBuffers);
        
                d_cacheddb = existingGpuDBAllocation;

                if(d_cacheddb->getNumChars() >= numSubjectBytes && d_cacheddb->getNumSubjects() >= numSubjects){
                    numBatchesInCachedDB = dbBatches.size();
                    charsOfBatches = numSubjectBytes;
                    subjectsOfBatches = numSubjects;

                    for(int i = 0; i < numCopyBuffers; i++){
                        if(needsPinnedStagingBuffers){
                            h_chardata_vec[i].resize(maxBatchBytes);
                            h_lengthdata_vec[i].resize(maxBatchSequences);
                            h_offsetdata_vec[i].resize(maxBatchSequences+1);
                        }
                        pinnedBufferEvents[i] = CudaEvent{cudaEventDisableTiming}; CUERR;
                        deviceBufferEvents[i] = CudaEvent{cudaEventDisableTiming}; CUERR;
                        //d_overflow_positions_vec[i].resize(numSubjects);
                    }
                }else{
                    //allocate a double buffer for batch transfering
                    size_t memoryRequiredForBatchedProcessing = 0;
                    memoryRequiredForBatchedProcessing += maxBatchBytes * numCopyBuffers; // d_chardata_vec
                    memoryRequiredForBatchedProcessing += sizeof(SequenceLengthT) * maxBatchSequences * numCopyBuffers; //d_lengthdata_vec
                    memoryRequiredForBatchedProcessing += sizeof(size_t) * (maxBatchSequences+1) * numCopyBuffers; //d_offsetdata_vec
                    usedGpuMem += memoryRequiredForBatchedProcessing;

                    //std::cout << "usedGpuMem " << usedGpuMem << ", gpumemlimit " << gpumemlimit << "\n";

                    //cached db is already accounted for because gpumemlimit was obtained after cached db was allocated

                    // if(usedGpuMem > gpumemlimit){
                    //     throw std::runtime_error("Out of memory working set");
                    // }
                    
                    for(int i = 0; i < numCopyBuffers; i++){
                        if(needsPinnedStagingBuffers){
                            h_chardata_vec[i].resize(maxBatchBytes);
                            h_lengthdata_vec[i].resize(maxBatchSequences);
                            h_offsetdata_vec[i].resize(maxBatchSequences+1);
                        }
                        d_chardata_vec[i].resize(maxBatchBytes);
                        d_lengthdata_vec[i].resize(maxBatchSequences);
                        d_offsetdata_vec[i].resize(maxBatchSequences+1);
                        pinnedBufferEvents[i] = CudaEvent{cudaEventDisableTiming}; CUERR;
                        deviceBufferEvents[i] = CudaEvent{cudaEventDisableTiming}; CUERR;
                        //d_overflow_positions_vec[i].resize(numSubjects);
                    }

                    //count how many batches fit into d_cacheddb

                    numBatchesInCachedDB = 0;
                    charsOfBatches = 0;
                    subjectsOfBatches = 0;
                    for(; numBatchesInCachedDB < dbBatches.size(); numBatchesInCachedDB++){
                        const auto& batch = dbBatches[numBatchesInCachedDB];
                        if(subjectsOfBatches + batch.usedSeq <= d_cacheddb->getNumSubjects() && charsOfBatches + batch.usedBytes <= d_cacheddb->getNumChars()){
                            //ok, fits
                            charsOfBatches += batch.usedBytes;
                            subjectsOfBatches += batch.usedSeq;
                        }else{
                            //does not fit
                            break;
                        }
                    }
                    assert(charsOfBatches <= d_cacheddb->getNumChars());
                    assert(subjectsOfBatches <= d_cacheddb->getNumSubjects());
                    assert(numBatchesInCachedDB < dbBatches.size());


                }
            }
                
            void setPartitionOffsets(const HostGpuPartitionOffsets& offsets){
                deviceGpuPartitionOffsets = DeviceGpuPartitionOffsets(offsets);
            }
            
            size_t getNumCharsInCachedDB() const{
                return charsOfBatches;
            }

            size_t getNumSequencesInCachedDB() const{
                return subjectsOfBatches;
            }

            size_t getNumBatchesInCachedDB() const{
                return numBatchesInCachedDB;
            }

            int deviceId;
            int numCopyBuffers;
            int copyBufferIndex = 0;
            size_t numBatchesInCachedDB = 0;
            size_t charsOfBatches = 0;
            size_t subjectsOfBatches = 0;

            CudaStream hostFuncStream;
            CudaEvent forkStreamEvent;

            std::shared_ptr<GpuDatabaseAllocationBase> d_cacheddb;

            
            std::vector<MyPinnedBuffer<char>> h_chardata_vec;
            std::vector<MyPinnedBuffer<SequenceLengthT>> h_lengthdata_vec;
            std::vector<MyPinnedBuffer<size_t>> h_offsetdata_vec;
            std::vector<MyDeviceBuffer<char>> d_chardata_vec;
            std::vector<MyDeviceBuffer<SequenceLengthT>> d_lengthdata_vec;
            std::vector<MyDeviceBuffer<size_t>> d_offsetdata_vec;
            std::vector<CudaStream> copyStreams;
            std::vector<CudaEvent> pinnedBufferEvents;
            std::vector<CudaEvent> deviceBufferEvents;
        
            DeviceGpuPartitionOffsets deviceGpuPartitionOffsets;
        };

        struct SequenceLengthStatistics{
            SequenceLengthT max_length = 0;
            SequenceLengthT min_length = std::numeric_limits<SequenceLengthT>::max();
            size_t sumOfLengths = 0;
        };
    
        struct MemoryConfig{
            size_t maxBatchBytes = 128ull * 1024ull * 1024ull;
            size_t maxBatchSequences = 10'000'000;
            size_t maxGpuMem = std::numeric_limits<size_t>::max();
        };

    private:
        struct BatchDstInfo{
            bool isUploaded{};
            char* charsPtr{};
            SequenceLengthT* lengthsPtr{};
            size_t* offsetsPtr{};
        };

    public:

        GpuDatabase(
            std::vector<int> deviceIds_, 
            const MemoryConfig& memoryConfig_,
            bool verbose_
        ) : verbose(verbose_), memoryConfig(memoryConfig_), deviceIds(std::move(deviceIds_))
        {
            if(deviceIds.size() == 0){ 
                throw std::runtime_error("No device selected");
            
            }
            RevertDeviceId rdi{};

            initializeGpus();

            dbIsReady = false;
        }

        GpuDatabase() = delete;
        GpuDatabase(const GpuDatabase&) = delete;
        GpuDatabase(GpuDatabase&&) = default;
        GpuDatabase& operator=(const GpuDatabase&) = delete;
        GpuDatabase& operator=(GpuDatabase&&) = default;

        const std::vector<int>& getDeviceIds() const{
            return deviceIds;
        }


        void setDatabase(std::shared_ptr<DB> dbPtr){
            RevertDeviceId rdi{};
            fullHostDB = AnyDBWrapper(dbPtr);
            makeReady();
        }

        void setDatabase(std::shared_ptr<DBWithVectors> dbPtr){
            RevertDeviceId rdi{};
            fullHostDB = AnyDBWrapper(dbPtr);
            makeReady();
        }

        void setDatabase(std::shared_ptr<PseudoDB> dbPtr){
            RevertDeviceId rdi{};
            fullHostDB = AnyDBWrapper(dbPtr);
            makeReady();
        }

        void setDatabase(std::shared_ptr<MMseqsDB> dbPtr){
            RevertDeviceId rdi{};
            fullHostDB = AnyDBWrapper(dbPtr);
            makeReady();
        }

        void setDatabase(std::shared_ptr<ExternalDB> dbPtr){
            RevertDeviceId rdi{};
            fullHostDB = AnyDBWrapper(dbPtr);
            makeReady();
        }

        void setDatabase(std::shared_ptr<DB> dbPtr, const std::vector<std::shared_ptr<GpuDatabaseAllocationBase>>& existingFullGpuDBAllocations){
            RevertDeviceId rdi{};
            fullHostDB = AnyDBWrapper(dbPtr);
            makeReadyWithExistingFullGpuDB(existingFullGpuDBAllocations);
        }

        void setDatabase(std::shared_ptr<DBWithVectors> dbPtr, const std::vector<std::shared_ptr<GpuDatabaseAllocationBase>>& existingFullGpuDBAllocations){
            RevertDeviceId rdi{};
            fullHostDB = AnyDBWrapper(dbPtr);
            makeReadyWithExistingFullGpuDB(existingFullGpuDBAllocations);
        }

        void setDatabase(std::shared_ptr<PseudoDB> dbPtr, const std::vector<std::shared_ptr<GpuDatabaseAllocationBase>>& existingFullGpuDBAllocations){
            RevertDeviceId rdi{};
            fullHostDB = AnyDBWrapper(dbPtr);
            makeReadyWithExistingFullGpuDB(existingFullGpuDBAllocations);
        }

        void setDatabase(std::shared_ptr<MMseqsDB> dbPtr, const std::vector<std::shared_ptr<GpuDatabaseAllocationBase>>& existingFullGpuDBAllocations){
            RevertDeviceId rdi{};
            fullHostDB = AnyDBWrapper(dbPtr);
            makeReadyWithExistingFullGpuDB(existingFullGpuDBAllocations);
        }

        void setDatabase(std::shared_ptr<ExternalDB> dbPtr, const std::vector<std::shared_ptr<GpuDatabaseAllocationBase>>& existingFullGpuDBAllocations){
            RevertDeviceId rdi{};
            fullHostDB = AnyDBWrapper(dbPtr);
            makeReadyWithExistingFullGpuDB(existingFullGpuDBAllocations);
        }

        std::vector<std::shared_ptr<GpuDatabaseAllocationBase>> getFullGpuDBAllocations(){
            if(!dbIsReady) return {};            

            prefetchDBToGpus();

            std::vector<std::shared_ptr<GpuDatabaseAllocationBase>> result;

            const int numGpus = deviceIds.size();
            for(int gpu = 0; gpu < numGpus; gpu++){
                auto& ws = *workingSets[gpu];
                
                result.push_back(ws.d_cacheddb);
            }

            return result;
        }

        std::string_view getReferenceHeader(ReferenceIdT referenceId) const{
            const auto& data = fullHostDB.getData();
            const char* const headerBegin = data.headers() + data.headerOffsets()[referenceId];
            const char* const headerEnd = data.headers() + data.headerOffsets()[referenceId+1];
            return std::string_view(headerBegin, std::distance(headerBegin, headerEnd));
        }

        int getReferenceLength(ReferenceIdT referenceId) const{
            const auto& data = fullHostDB.getData();
            return data.lengths()[referenceId];
        }

        std::string getReferenceSequence(ReferenceIdT referenceId) const{
            const auto& data = fullHostDB.getData();
            const char* const begin = data.chars() + data.offsets()[referenceId];
            const char* const end = begin + getReferenceLength(referenceId);

            std::string sequence(end - begin, '\0');
            std::transform(
                begin, 
                end,
                sequence.begin(),
                InverseConvertAA_20{}
            );

            return sequence;
        }

        const char* getReferenceDatabaseSequenceHostPtr(ReferenceIdT referenceId){
            assert(referenceId < getNumSequences());
            
            const auto& data = fullHostDB.getData();
            const char* begin = data.chars() + data.offsets()[referenceId];
            return begin;
        }

        void prefetchDBToGpus(){
            nvtx::ScopedRange sr("prefetchDBToGpus", 1);
            RevertDeviceId rdi{};

            const int numGpus = deviceIds.size();
            std::vector<int> copyIds;

            helpers::CpuTimer copyTimer("transfer DB to GPUs");
            for(int gpu = 0; gpu < numGpus; gpu++){
                CUDACHECK(cudaSetDevice(deviceIds[gpu]));
                auto& ws = *workingSets[gpu];

                if(ws.getNumBatchesInCachedDB() > 0 && !batchPlansDstInfoVec_cachedDB[gpu][0].isUploaded){
                    const auto& plan = batchPlans_cachedDB[gpu][0];
                    const int currentBuffer = 0;
                    cudaStream_t H2DcopyStream = ws.copyStreams[currentBuffer];

                    executeCopyPlanH2DDirect(
                        plan,
                        ws.d_cacheddb->getCharData(),
                        ws.d_cacheddb->getLengthData(),
                        ws.d_cacheddb->getOffsetData(),
                        subPartitionsForGpus[gpu],
                        H2DcopyStream
                    );
                    
                    copyIds.push_back(gpu);

                    markCachedDBBatchesAsUploaded(gpu);
                }
            }
            for(int gpu : copyIds){
                CUDACHECK(cudaSetDevice(deviceIds[gpu]));
                CUDACHECK(cudaDeviceSynchronize());
            }
            copyTimer.stop();
            if(copyIds.size() > 0){
                if(verbose){
                    std::cout << "Transferred DB data in advance to GPU(s) ";
                    for(auto x : copyIds){
                        std::cout << x << " ";
                    }
                    std::cout << "\n";
                    copyTimer.print();
                }
            }
        }

        size_t getNumSequences() const{
            return fullHostDB.getData().numSequences();
        }

        void printDBInfo() const{
            nvtx::ScopedRange sr("printDBInfo", 0);
            const size_t numSequences = getNumSequences();
            std::cout << numSequences << " sequences, " << fullHostDB.getData().numChars() << " characters\n";

            SequenceLengthStatistics stats = getSequenceLengthStatistics();

            std::cout << "Min length " << stats.min_length << ", max length " << stats.max_length 
                << ", avg length " << stats.sumOfLengths / numSequences << "\n";
        }

        void printDBLengthPartitions() const{
            auto lengthBoundaries = getLengthPartitionBoundaries();
            const int numLengthPartitions = getLengthPartitionBoundaries().size();

            for(int i = 0; i < numLengthPartitions; i++){
                std::cout << "<= " << lengthBoundaries[i] << ": " << fullHostDB_numSequencesPerLengthPartition[i] << "\n";
            }
        }


        //streams[i] belongs to gpu deviceIds[i]
        void forEachBatchOfSequencesInDatabase(
            ForEachCallback&& callback, 
            cuda::std::span<cudaStream_t> streams,
            bool processCachedDBInSeparateBatches
        ){
            nvtx::ScopedRange sr("forEachBatchOfSequencesInDatabase", 0);

            // std::cout << "ProcessQueryOnGpus: dstinfos isUploaded\n";
            // for(size_t i = 0; i < batchPlans[0].size(); i++){
            //     std::cout << batchPlansDstInfoVec[0][i].isUploaded << " ";
            // }
            // std::cout << "\n";

            const std::vector<std::vector<DBdataView>>& dbPartitionsPerGpu = subPartitionsForGpus;

            // constexpr auto boundaries = getLengthPartitionBoundaries();
            // constexpr int numLengthPartitions = boundaries.size();
            const int numGpus = deviceIds.size();
            const bool useExtraThreadForBatchTransfer = shouldUseHostFuncForTransfer();
        
            const size_t totalNumberOfSequencesToProcess = std::accumulate(numSequencesPerGpu.begin(), numSequencesPerGpu.end(), 0u);
            
            size_t totalNumberOfProcessedSequences = 0;    
        
            //variables per gpu to keep between loops
            struct Variables{
                int currentBuffer = 0;
                int previousBuffer = 0;
                cudaStream_t H2DcopyStream = cudaStreamLegacy;
                char* h_inputChars = nullptr;
                SequenceLengthT* h_inputLengths = nullptr;
                size_t* h_inputOffsets = nullptr;
                char* d_inputChars = nullptr;
                SequenceLengthT* d_inputLengths = nullptr;
                size_t* d_inputOffsets = nullptr;
                //int* d_overflow_number = nullptr;
                //ReferenceIdT* d_overflow_positions = nullptr;
                const std::vector<DeviceBatchCopyToPinnedPlan>* batchPlansPtr;
                const std::vector<DeviceBatchCopyToPinnedPlan>* batchPlansCachedDBPtr;
                const DeviceBatchCopyToPinnedPlan* currentPlanPtr;
                size_t processedSequences = 0;
                size_t processedBatches = 0;
            };
        
            std::vector<Variables> variables_vec(numGpus);
            //init variables
            for(int gpu = 0; gpu < numGpus; gpu++){
                CUDACHECK(cudaSetDevice(deviceIds[gpu]));
                const auto& ws = *workingSets[gpu];
                auto& variables = variables_vec[gpu];
                variables.processedSequences = 0;
                variables.processedBatches = 0;
                variables.batchPlansPtr = &batchPlans[gpu];
                variables.batchPlansCachedDBPtr = &batchPlans_cachedDB[gpu];
            }
            
            while(totalNumberOfProcessedSequences < totalNumberOfSequencesToProcess){
                //set up gpu variables for current iteration
                for(int gpu = 0; gpu < numGpus; gpu++){
                    CUDACHECK(cudaSetDevice(deviceIds[gpu]));
                    auto& ws = *workingSets[gpu];
                    auto& variables = variables_vec[gpu];
                    if(variables.processedBatches < variables.batchPlansPtr->size()){
                        // std::cout << "batch " << variables.processedBatches << "\n";
                        // CUDACHECK(cudaDeviceSynchronize());

                        if(variables.processedBatches < ws.getNumBatchesInCachedDB()){
                            //will process a batch that could be cached in gpu memory
                            if(batchPlansDstInfoVec[gpu][variables.processedBatches].isUploaded == false){
                                //it is not cached, need upload
                                variables.currentBuffer = ws.copyBufferIndex;
                                if(variables.currentBuffer == 0){
                                    variables.previousBuffer = ws.numCopyBuffers - 1;
                                }else{
                                    variables.previousBuffer = (variables.currentBuffer - 1);
                                } 
                                variables.H2DcopyStream = ws.copyStreams[variables.currentBuffer];
                                if(ws.h_chardata_vec[variables.currentBuffer].size() > 0){
                                    variables.h_inputChars = ws.h_chardata_vec[variables.currentBuffer].data();
                                }else{
                                    variables.h_inputChars = nullptr;
                                }
                                if(ws.h_lengthdata_vec[variables.currentBuffer].size() > 0){
                                    variables.h_inputLengths = ws.h_lengthdata_vec[variables.currentBuffer].data();
                                }else{
                                    variables.h_inputLengths = nullptr;
                                }
                                if(ws.h_offsetdata_vec[variables.currentBuffer].size() > 0){
                                    variables.h_inputOffsets = ws.h_offsetdata_vec[variables.currentBuffer].data();
                                }else{
                                    variables.h_inputOffsets = nullptr;
                                }
                                variables.d_inputChars = batchPlansDstInfoVec[gpu][variables.processedBatches].charsPtr;
                                variables.d_inputLengths = batchPlansDstInfoVec[gpu][variables.processedBatches].lengthsPtr;
                                variables.d_inputOffsets = batchPlansDstInfoVec[gpu][variables.processedBatches].offsetsPtr;
                                //variables.d_overflow_number = ws.d_overflow_number.data() + variables.currentBuffer;
                                //variables.d_overflow_positions = ws.d_overflow_positions_vec[variables.currentBuffer].data();
                            }else{
                                if(processCachedDBInSeparateBatches){
                                    /*
                                    !singleTile -> multiTile, requires temporary storage.
                                    process only this batch, not as part of the cached DB.
                                    this allows to better restrict the maximum subject length in the batch which in turn 
                                    reduces the amount of temporary storage necessary to saturate all SMs for batches with short subjects.
                                    */
                                    variables.currentBuffer = 0;
                                    variables.previousBuffer = 0;
                                    variables.H2DcopyStream = ws.copyStreams[0];
                                    variables.h_inputChars = nullptr;
                                    variables.h_inputLengths = nullptr;
                                    variables.h_inputOffsets = nullptr;
                                    variables.d_inputChars = batchPlansDstInfoVec[gpu][variables.processedBatches].charsPtr;
                                    variables.d_inputLengths = batchPlansDstInfoVec[gpu][variables.processedBatches].lengthsPtr;
                                    variables.d_inputOffsets = batchPlansDstInfoVec[gpu][variables.processedBatches].offsetsPtr;
                                }else{
                                    //already uploaded. process all batches for cached db together
                                    assert(variables.processedBatches == 0);
                                    variables.currentBuffer = 0;
                                    variables.previousBuffer = 0;
                                    variables.H2DcopyStream = ws.copyStreams[0];
                                    variables.h_inputChars = nullptr;
                                    variables.h_inputLengths = nullptr;
                                    variables.h_inputOffsets = nullptr;
                                    variables.d_inputChars = ws.d_cacheddb->getCharData();
                                    variables.d_inputLengths = ws.d_cacheddb->getLengthData();
                                    variables.d_inputOffsets = ws.d_cacheddb->getOffsetData();
                                }
                            }
                        }else{
                            //will process batch that cannot be cached
                            //upload to double buffer
                            variables.currentBuffer = ws.copyBufferIndex;
                            if(variables.currentBuffer == 0){
                                variables.previousBuffer = ws.numCopyBuffers - 1;
                            }else{
                                variables.previousBuffer = (variables.currentBuffer - 1);
                            } 
                            variables.H2DcopyStream = ws.copyStreams[variables.currentBuffer];
                            if(ws.h_chardata_vec[variables.currentBuffer].size() > 0){
                                variables.h_inputChars = ws.h_chardata_vec[variables.currentBuffer].data();
                            }else{
                                variables.h_inputChars = nullptr;
                            }
                            if(ws.h_lengthdata_vec[variables.currentBuffer].size() > 0){
                                variables.h_inputLengths = ws.h_lengthdata_vec[variables.currentBuffer].data();
                            }else{
                                variables.h_inputLengths = nullptr;
                            }
                            if(ws.h_offsetdata_vec[variables.currentBuffer].size() > 0){
                                variables.h_inputOffsets = ws.h_offsetdata_vec[variables.currentBuffer].data();
                            }else{
                                variables.h_inputOffsets = nullptr;
                            }
                            variables.d_inputChars = batchPlansDstInfoVec[gpu][variables.processedBatches].charsPtr;
                            variables.d_inputLengths = batchPlansDstInfoVec[gpu][variables.processedBatches].lengthsPtr;
                            variables.d_inputOffsets = batchPlansDstInfoVec[gpu][variables.processedBatches].offsetsPtr;
                        }
                    }
                }
                //upload batch
                for(int gpu = 0; gpu < numGpus; gpu++){
                    CUDACHECK(cudaSetDevice(deviceIds[gpu]));
                    // CUDACHECK(cudaDeviceSynchronize());
                    auto& ws = *workingSets[gpu];
                    auto& variables = variables_vec[gpu];
                    if(variables.processedBatches < variables.batchPlansPtr->size()){
                        const bool needsUpload = !batchPlansDstInfoVec[gpu][variables.processedBatches].isUploaded;

                        variables.currentPlanPtr = [&](){
                            if(processCachedDBInSeparateBatches){
                                return &(*variables.batchPlansPtr)[variables.processedBatches];
                            }else{
                                if(variables.processedBatches < ws.getNumBatchesInCachedDB()){
                                    if(!needsUpload){
                                        return &(*variables.batchPlansCachedDBPtr)[0];
                                        //return &(*variables.batchPlansPtr)[variables.processedBatches];
                                    }else{
                                        return &(*variables.batchPlansPtr)[variables.processedBatches];
                                    }
                                }else{
                                    return &(*variables.batchPlansPtr)[variables.processedBatches];
                                }
                            }
                        }();
                            
        
                        if(needsUpload){
                            //transfer data
                            //can only overwrite device buffer if it is no longer in use on workstream
                            CUDACHECK(cudaStreamWaitEvent(variables.H2DcopyStream, ws.deviceBufferEvents[variables.currentBuffer], 0));
                            // CUDACHECK(cudaDeviceSynchronize());
                            if(useExtraThreadForBatchTransfer){
                                assert(variables.h_inputChars != nullptr);
                                assert(variables.h_inputLengths != nullptr);
                                assert(variables.h_inputOffsets != nullptr);

                                CUDACHECK(cudaStreamWaitEvent(ws.hostFuncStream, ws.pinnedBufferEvents[variables.currentBuffer]));
                                executePinnedCopyPlanWithHostCallback(
                                    *variables.currentPlanPtr, 
                                    variables.h_inputChars,
                                    variables.h_inputLengths,
                                    variables.h_inputOffsets,
                                    dbPartitionsPerGpu[gpu], 
                                    ws.hostFuncStream
                                );
                                CUDACHECK(cudaEventRecord(ws.forkStreamEvent, ws.hostFuncStream));
                                CUDACHECK(cudaStreamWaitEvent(variables.H2DcopyStream, ws.forkStreamEvent, 0));
        
                                CUDACHECK(cudaMemcpyAsync(
                                    variables.d_inputChars,
                                    variables.h_inputChars,
                                    variables.currentPlanPtr->usedBytes,
                                    H2D,
                                    variables.H2DcopyStream
                                ));
                                CUDACHECK(cudaMemcpyAsync(
                                    variables.d_inputLengths,
                                    variables.h_inputLengths,
                                    sizeof(SequenceLengthT) * variables.currentPlanPtr->usedSeq,
                                    H2D,
                                    variables.H2DcopyStream
                                ));
                                CUDACHECK(cudaMemcpyAsync(
                                    variables.d_inputOffsets,
                                    variables.h_inputOffsets,
                                    sizeof(size_t) * (variables.currentPlanPtr->usedSeq+1),
                                    H2D,
                                    variables.H2DcopyStream
                                ));
                                CUDACHECK(cudaEventRecord(ws.pinnedBufferEvents[variables.currentBuffer], variables.H2DcopyStream));
                            }else{
                                //synchronize to avoid overwriting pinned buffer of target before it has been fully transferred
                                // CUDACHECK(cudaEventSynchronize(ws.pinnedBufferEvents[variables.currentBuffer]));
                                // CUDACHECK(cudaDeviceSynchronize());
                                executeCopyPlanH2DDirect(
                                    *variables.currentPlanPtr, 
                                    variables.d_inputChars,
                                    variables.d_inputLengths,
                                    variables.d_inputOffsets,
                                    dbPartitionsPerGpu[gpu], 
                                    variables.H2DcopyStream
                                );

                                // CUDACHECK(cudaDeviceSynchronize());
                                // assert(variables.h_inputChars != nullptr);
                                // assert(variables.h_inputLengths != nullptr);
                                // assert(variables.h_inputOffsets != nullptr);
        
                                // executePinnedCopyPlanSerialAndTransferToGpu(
                                //     *variables.currentPlanPtr, 
                                //     variables.h_inputChars,
                                //     variables.h_inputLengths,
                                //     variables.h_inputOffsets,
                                //     variables.d_inputChars,
                                //     variables.d_inputLengths,
                                //     variables.d_inputOffsets,
                                //     dbPartitionsPerGpu[gpu], 
                                //     variables.H2DcopyStream
                                // );

                                // CUDACHECK(cudaEventRecord(ws.pinnedBufferEvents[variables.currentBuffer], variables.H2DcopyStream));
                            }
                            
                            //ensure copy is complete before using the data in streams[gpu]
                            CUDACHECK(cudaEventRecord(ws.forkStreamEvent, variables.H2DcopyStream));
                            CUDACHECK(cudaStreamWaitEvent(streams[gpu], ws.forkStreamEvent, 0));
                        }
                    }
                }


                //determine maximum number of sequences to process over all gpus
                size_t maxNumSequencesInBatchForGpus = 0;
                for(int gpu = 0; gpu < numGpus; gpu++){
                    auto& variables = variables_vec[gpu];
                    if(variables.processedBatches < variables.batchPlansPtr->size()){
                        maxNumSequencesInBatchForGpus = std::max(maxNumSequencesInBatchForGpus, variables.currentPlanPtr->usedSeq);
                    }
                }

                std::vector<CallbackInputPerGpu> callbackInput;
                for(int gpu = 0; gpu < numGpus; gpu++){
                    auto& ws = *workingSets[gpu];
                    auto& variables = variables_vec[gpu];
                    CallbackInputPerGpu data;
                    data.deviceId = deviceIds[gpu];
                    data.d_inputChars = variables.d_inputChars;
                    data.d_inputLengths = variables.d_inputLengths;
                    const bool convertToZeroBased = processCachedDBInSeparateBatches;
                    data.d_inputOffsets = makeCustomOffsetIterator(variables.d_inputOffsets, convertToZeroBased);
                    data.stream = streams[gpu];

                    if(variables.processedBatches < variables.batchPlansPtr->size()){
                        data.numInputSequences = variables.currentPlanPtr->usedSeq;
                        data.numInputChars = variables.currentPlanPtr->usedBytes;

                        auto dbview = fullHostDB.getData(); 
                        data.sequenceLengthUpperBound = dbview.lengths()[dbview.numSequences()-1];
                        assert(variables.currentPlanPtr->h_partitionIds.size() > 0);
                        const int maxLengthPartitionId = *std::max_element(variables.currentPlanPtr->h_partitionIds.begin(), variables.currentPlanPtr->h_partitionIds.end());
                        auto lengthBoundaries = getLengthPartitionBoundaries();
                        if(maxLengthPartitionId < int(lengthBoundaries.size())-1){
                            data.sequenceLengthUpperBound = lengthBoundaries[maxLengthPartitionId];
                        }
                    }else{
                        data.numInputSequences = 0;
                        data.numInputChars = 0;
                        data.sequenceLengthUpperBound = 0;
                    }
                    callbackInput.push_back(data);
                }
                // CUDACHECK(cudaDeviceSynchronize());
                callback(cuda::std::span(callbackInput.begin(), callbackInput.end()));
                // CUDACHECK(cudaDeviceSynchronize());
        
                //finish processing of batch
                for(int gpu = 0; gpu < numGpus; gpu++){
                    CUDACHECK(cudaSetDevice(deviceIds[gpu]));
                    auto& ws = *workingSets[gpu];
                    const auto& variables = variables_vec[gpu];
                    if(variables.processedBatches < variables.batchPlansPtr->size()){                
                        //the batch is done and its data can be resused
                        CUDACHECK(cudaEventRecord(ws.deviceBufferEvents[variables.currentBuffer], streams[gpu]));
                
                        ws.copyBufferIndex = (ws.copyBufferIndex+1) % ws.numCopyBuffers;
                    }
                }
                // CUDACHECK(cudaDeviceSynchronize());
                //update running numbers
                for(int gpu = 0; gpu < numGpus; gpu++){
                    auto& variables = variables_vec[gpu];
                    if(variables.processedBatches < variables.batchPlansPtr->size()){

                        variables.processedSequences += variables.currentPlanPtr->usedSeq;
                        if(processCachedDBInSeparateBatches){
                            variables.processedBatches++;
                        }else{
                            if(batchPlansDstInfoVec[gpu][variables.processedBatches].isUploaded){
                                variables.processedBatches += workingSets[gpu]->getNumBatchesInCachedDB();                            
                                //variables.processedBatches++;
                            }else{
                                variables.processedBatches++;
                            }
                        }
                        //std::cout << "variables.processedBatches: " << variables.processedBatches << "\n";
        
                        totalNumberOfProcessedSequences += variables.currentPlanPtr->usedSeq;
                    } 
                }
        
            } //while not done
        
        
            for(int gpu = 0; gpu < numGpus; gpu++){
                CUDACHECK(cudaSetDevice(deviceIds[gpu]));
                auto& ws = *workingSets[gpu];

                if(batchPlansDstInfoVec[gpu].size() > 0){
                    if(!batchPlansDstInfoVec[gpu][0].isUploaded){
                        //all batches for cached db are now resident in gpu memory. update the flags
                        if(ws.getNumBatchesInCachedDB() > 0){
                            markCachedDBBatchesAsUploaded(gpu);

                            // current offsets in cached db store the offsets for each batch, i.e. for each batch the offsets will start again at 0
                            // compute prefix sum to obtain the single-batch offsets
                
                            CUDACHECK(cudaMemsetAsync(ws.d_cacheddb->getOffsetData(), 0, sizeof(size_t), streams[gpu]));
                
                            auto d_paddedLengths = thrust::make_transform_iterator(
                                ws.d_cacheddb->getLengthData(),
                                RoundToNextMultiple<size_t, 4>{}
                            );
                
                            thrust::inclusive_scan(
                                thrust::cuda::par_nosync(thrust_async_allocator<char>(streams[gpu])).on(streams[gpu]),
                                d_paddedLengths,
                                d_paddedLengths + ws.getNumSequencesInCachedDB(),
                                ws.d_cacheddb->getOffsetData() + 1
                            );
                        }
                    }
                }
            }

            // epilogueFunction(this, streams);
        }


        //streams[i] belongs to gpu deviceIds[i]
        void forEachBatchOfSequencesInDatabaseSubset(
            ForEachCallback&& callback, 
            cuda::std::span<ReferenceIdT> sortedSubsetRefIds,
            cuda::std::span<cudaStream_t> streams
        ){
            nvtx::ScopedRange sr("forEachBatchOfSequencesInDatabaseSubset", 0);

            if(sortedSubsetRefIds.size() == 0){
                return;
            }

            assert(std::is_sorted(sortedSubsetRefIds.begin(), sortedSubsetRefIds.end()));

            //todo: for proper multi-gpu we need to find the correct gpu for each target subject to re-use the cached data
            //for now, multi-gpu will always gather the data on the host and transfer it to gpu 0
            const int numGpus = deviceIds.size();

            prefetchDBToGpus();

            // std::vector<ReferenceIdT> sortedSubsetRefIds(subsetRefIds.begin(), subsetRefIds.end());
            // std::sort(sortedSubsetRefIds.begin(), sortedSubsetRefIds.end());
            const size_t numCachedSubjects = workingSets[0]->d_cacheddb->getNumSubjects();
            auto cachedTargetSubjectIdsEnd = sortedSubsetRefIds.begin();

            if(numGpus == 1){
                cachedTargetSubjectIdsEnd = std::lower_bound(sortedSubsetRefIds.begin(), sortedSubsetRefIds.end(), numCachedSubjects);
            }
            const size_t numCachedTargetSubjects = std::distance(sortedSubsetRefIds.begin(), cachedTargetSubjectIdsEnd);
            const size_t numUncachedTargetSubjects = std::distance(cachedTargetSubjectIdsEnd, sortedSubsetRefIds.end());
            if(verbose){
                std::cout << "numCachedTargetSubjects " << numCachedTargetSubjects << "\n";
                std::cout << "numUncachedTargetSubjects " << numUncachedTargetSubjects << "\n";
            }

            //process subjects which reside in gpu memory
            if(numCachedTargetSubjects > 0){
                assert(numGpus == 1);
                cudaStream_t stream = streams[0];
                auto& ws = *workingSets[0];

                const size_t maxbatchsize = memoryConfig.maxBatchSequences;

#if defined(__HIPCC__)
                thrust::device_vector<ReferenceIdT> d_subsetRefIds;
                thrust::device_vector<SequenceLengthT> d_subsetLengths;
                thrust::device_vector<size_t> d_subsetOffsets;
                d_subsetRefIds.resize(maxbatchsize);
                d_subsetLengths.resize(maxbatchsize);
                d_subsetOffsets.resize(maxbatchsize);
#else
                thrust::device_vector<ReferenceIdT, thrust_async_allocator<ReferenceIdT>> d_subsetRefIds{thrust_async_allocator<ReferenceIdT>(stream)};
                thrust::device_vector<SequenceLengthT, thrust_async_allocator<SequenceLengthT>> d_subsetLengths{thrust_async_allocator<SequenceLengthT>(stream)};
                thrust::device_vector<size_t, thrust_async_allocator<size_t>> d_subsetOffsets{thrust_async_allocator<size_t>(stream)};
                d_subsetRefIds.resize(maxbatchsize, thrust::no_init);
                d_subsetLengths.resize(maxbatchsize, thrust::no_init);
                d_subsetOffsets.resize(maxbatchsize, thrust::no_init);
#endif


                for(size_t index = 0; index < numCachedTargetSubjects; index += maxbatchsize){
                    const int numInBatch = std::min(numCachedTargetSubjects - index, maxbatchsize);
                    CUDACHECK(cudaMemcpyAsync(
                        d_subsetRefIds.data().get(),
                        sortedSubsetRefIds.data() + index,
                        sizeof(ReferenceIdT) * numInBatch,
                        cudaMemcpyHostToDevice,
                        stream
                    ));
                    // {
                    //     std::cout << "d_subsetRefIds\n";
                    //     for(int i = 0; i < numInBatch; i++){
                    //         std::cout << d_subsetRefIds[i] << " ";
                    //     }
                    //     std::cout << "\n";

                    //     std::cout << "expected lengths:\n";
                    //     auto dbview = fullHostDB.getData(); 
                    //     for(int i = 0; i < numInBatch; i++){
                    //         std::cout << dbview.lengths()[d_subsetRefIds[i]] << " ";
                    //     }
                    //     std::cout << "\n";
                    // }
                    thrust::gather(
                        thrust::cuda::par_nosync.on(stream),
                        d_subsetRefIds.begin(),
                        d_subsetRefIds.begin() + numInBatch,
                        thrust::make_zip_iterator(
                            ws.d_cacheddb->getLengthData(),
                            ws.d_cacheddb->getOffsetData()
                        ),
                        thrust::make_zip_iterator(
                            d_subsetLengths.begin(),
                            d_subsetOffsets.begin()
                        )
                    );

                    auto d_paddedLengths = thrust::make_transform_iterator(
                        d_subsetLengths.begin(),
                        RoundToNextMultiple<size_t, 4>{}
                    );
                    size_t batchBytes = thrust::reduce(
                        thrust::cuda::par_nosync(thrust_async_allocator<char>(streams[0])).on(streams[0]),
                        d_paddedLengths,
                        d_paddedLengths + numInBatch
                    );

                    std::vector<CallbackInputPerGpu> callbackInput;
                    for(int gpu = 0; gpu < numGpus; gpu++){
                        auto& ws = *workingSets[gpu];

                        CallbackInputPerGpu data;
                        data.deviceId = deviceIds[gpu];
                        data.d_inputChars = ws.d_cacheddb->getCharData();
                        data.d_inputLengths =  d_subsetLengths.data().get();
                        const bool convertToZeroBased = false;
                        data.d_inputOffsets = makeCustomOffsetIterator(d_subsetOffsets.data().get(), convertToZeroBased);
                        data.stream = streams[gpu];
                        data.numInputSequences = numInBatch;
                        data.numInputChars = batchBytes;

                        auto dbview = fullHostDB.getData(); 
                        data.sequenceLengthUpperBound = dbview.lengths()[dbview.numSequences()-1];

                        callbackInput.push_back(data);

                        //we are currently only using gpu 0
                        if(gpu > 0){
                            data.numInputSequences = 0;
                            data.numInputChars = 0;
                        }
                    }

                    callback(cuda::std::span(callbackInput.begin(), callbackInput.end()));

                }
               
                
            }

            //process subjects which reside in host memory
            if(numUncachedTargetSubjects > 0){
                auto& ws = *workingSets[0];

                cudaStream_t stream = streams[0];

                helpers::CpuTimer targetGatherTimer("targetGatherTimer");
                std::vector<char> targetChars;
                std::vector<size_t> targetOffsets(numUncachedTargetSubjects+1, 0);
                std::vector<SequenceLengthT> targetLengths(numUncachedTargetSubjects);
                const auto& data = fullHostDB.getData();
                for(size_t i = 0; i < numUncachedTargetSubjects; i++){
                    const ReferenceIdT subjectId = *(cachedTargetSubjectIdsEnd + i);
                    const size_t offsetBegin = data.offsets()[subjectId];
                    const size_t offsetEnd = data.offsets()[subjectId+1];
                    SequenceLengthT length = data.lengths()[subjectId];
                    targetChars.insert(targetChars.end(), data.chars() + offsetBegin, data.chars() + offsetEnd);
                    targetOffsets[i+1] = targetChars.size();
                    targetLengths[i] = length;
                }
                if(verbose){
                    targetGatherTimer.print();
                }

                std::vector<DBdataView> targetDBPartition{DBdataView(
                    0,
                    numUncachedTargetSubjects,
                    0,
                    targetChars.data(),
                    targetLengths.data(),
                    targetOffsets.data(),
                    nullptr,
                    nullptr
                )};
                std::vector<DeviceBatchCopyToPinnedPlan> targetBatchPlans = computeDbCopyPlan(
                    targetDBPartition,
                    {0},
                    memoryConfig.maxBatchBytes,
                    memoryConfig.maxBatchSequences
                );

                thrust::device_vector<char, thrust_async_allocator<char>> d_subsetChars{thrust_async_allocator<char>(stream)};
                thrust::device_vector<SequenceLengthT, thrust_async_allocator<SequenceLengthT>> d_subsetLengths{thrust_async_allocator<SequenceLengthT>(stream)};
                thrust::device_vector<size_t, thrust_async_allocator<size_t>> d_subsetOffsets{thrust_async_allocator<size_t>(stream)};

                d_subsetChars.resize(memoryConfig.maxBatchBytes, thrust::no_init);
                d_subsetLengths.resize(memoryConfig.maxBatchSequences, thrust::no_init);
                d_subsetOffsets.resize(memoryConfig.maxBatchSequences+1, thrust::no_init);

                for(const auto& batchPlan : targetBatchPlans){

                    executeCopyPlanH2DDirect(
                        batchPlan, 
                        d_subsetChars.data().get(),
                        d_subsetLengths.data().get(),
                        d_subsetOffsets.data().get(),
                        targetDBPartition, 
                        stream
                    );

                    std::vector<CallbackInputPerGpu> callbackInput;
                    for(int gpu = 0; gpu < numGpus; gpu++){
                        auto& ws = *workingSets[gpu];

                        CallbackInputPerGpu data;
                        data.deviceId = deviceIds[gpu];
                        data.d_inputChars = d_subsetChars.data().get();
                        data.d_inputLengths =  d_subsetLengths.data().get();
                        const bool convertToZeroBased = true;
                        data.d_inputOffsets = makeCustomOffsetIterator(d_subsetOffsets.data().get(), convertToZeroBased);
                        data.stream = streams[gpu];
                        data.numInputSequences = batchPlan.usedSeq;
                        data.numInputChars = batchPlan.usedBytes;

                        auto dbview = targetDBPartition[0]; 
                        data.sequenceLengthUpperBound = dbview.lengths()[dbview.numSequences()-1];
                        assert(batchPlan.h_partitionIds.size() > 0);
                        const int maxLengthPartitionId = *std::max_element(batchPlan.h_partitionIds.begin(), batchPlan.h_partitionIds.end());
                        auto lengthBoundaries = getLengthPartitionBoundaries();
                        if(maxLengthPartitionId < int(lengthBoundaries.size())-1){
                            data.sequenceLengthUpperBound = lengthBoundaries[maxLengthPartitionId];
                        }

                        //we are currently only using gpu 0
                        if(gpu > 0){
                            data.numInputSequences = 0;
                            data.numInputChars = 0;
                        }

                        callbackInput.push_back(data);
                    }

                    callback(cuda::std::span(callbackInput.begin(), callbackInput.end()));

                }
            }

        }

        SequenceLengthStatistics getSequenceLengthStatistics() const{
            if(dbSequenceLengthStatistics == nullptr){
                dbSequenceLengthStatistics = std::make_unique<SequenceLengthStatistics>();
                const auto& data = fullHostDB.getData();
                size_t numSeq = data.numSequences();

                for (size_t i=0; i < numSeq; i++) {
                    if (data.lengths()[i] > dbSequenceLengthStatistics->max_length) dbSequenceLengthStatistics->max_length = data.lengths()[i];
                    if (data.lengths()[i] < dbSequenceLengthStatistics->min_length) dbSequenceLengthStatistics->min_length = data.lengths()[i];
                    dbSequenceLengthStatistics->sumOfLengths += data.lengths()[i];
                }
            }
            return *dbSequenceLengthStatistics;
        }

        //transform per gpu local sequence indices into global sequence indices
        void convertLocalIndicesToGlobalIndices(
            int gpuPartion, 
            ReferenceIdT* d_indices,
            int size,
            cudaStream_t stream
        ){
            auto& ws = *workingSets[gpuPartion];
            transformLocalSequenceIndicesToGlobalIndicesKernel<<<SDIV(size, 128), 128, 0, stream>>>(
                gpuPartion,
                size,
                ws.deviceGpuPartitionOffsets.getDeviceView(),
                d_indices
            );
            CUDACHECKASYNC;
        }

    private:
        void initializeGpus(){
            const int numGpus = deviceIds.size();

            for(int i = 0; i < numGpus; i++){
                CUDACHECK(cudaSetDevice(deviceIds[i]));

                //TODO the application should set up the pool
                cudaMemPool_t mempool;
                CUDACHECK(cudaDeviceGetDefaultMemPool(&mempool, deviceIds[i]));
                uint64_t threshold = UINT64_MAX;
                CUDACHECK(cudaMemPoolSetAttribute(mempool, cudaMemPoolAttrReleaseThreshold, &threshold));
            }
        }


        // check if we can use GPU unified memory
        bool canUseHostDbInPlace() const {
            if(deviceIds.empty()){
                return false;
            }
            for(int id : deviceIds){
                int usesHostPageTables = 0;
                cudaDeviceGetAttribute(&usesHostPageTables, cudaDevAttrPageableMemoryAccessUsesHostPageTables, id);
                if(usesHostPageTables != 1){
                    return false;
                }
            }
            return true;
        }

        // check if DB fits in host memory
        bool hostDbFitsInMemory() const {
            const auto& data = fullHostDB.getData();
            const size_t dbBytes = data.numChars()
                + data.numSequences() * (sizeof(SequenceLengthT) + sizeof(size_t));
            const size_t reserve = size_t(4) << 30;
            RevertDeviceId rdi{};
            for(int id : deviceIds){
                cudaSetDevice(id);
                size_t freeMem = 0, totalMem = 0;
                cudaMemGetInfo(&freeMem, &totalMem);
                if(dbBytes + reserve > totalMem){
                    return false;
                }
            }
            return true;
        }

        void makeReady(){
            nvtx::ScopedRange sr("makeReady", 0);

            // don't duplicate memory on unified-memory GPUs, unless the database is too large
            if(canUseHostDbInPlace() && hostDbFitsInMemory()){
                const auto& data = fullHostDB.getData();
                std::vector<std::shared_ptr<GpuDatabaseAllocationBase>> hostViews(
                    deviceIds.size(),
                    std::make_shared<GpuDatabaseAllocationView>(
                        const_cast<char*>(data.chars()),
                        const_cast<SequenceLengthT*>(data.lengths()),
                        const_cast<size_t*>(data.offsets()),
                        data.numChars(),
                        data.numSequences()
                    )
                );
                makeReadyWithExistingFullGpuDB(hostViews);
                return;
            }

            dbSequenceLengthStatistics = nullptr;

            computeTotalNumSequencePerLengthPartition();
            partitionDBAmongstGpus();

            createDBBatchesForGpus();
            allocateGpuWorkingSets();
            assignBatchesToGpuMem();
            
            
            dbIsReady = true;
        }

        void makeReadyWithExistingFullGpuDB(const std::vector<std::shared_ptr<GpuDatabaseAllocationBase>>& existingFullGpuDBAllocations){
            nvtx::ScopedRange sr("makeReadyWithExistingFullGpuDB", 0);

            dbSequenceLengthStatistics = nullptr;

            computeTotalNumSequencePerLengthPartition();
            partitionDBAmongstGpus();

            createDBBatchesForGpus();
            allocateGpuWorkingSetsWithExistingFullGpuDB(existingFullGpuDBAllocations);
            assignBatchesToGpuMem();

            const int numGpus = deviceIds.size();
            for(int gpu = 0; gpu < numGpus; gpu++){
                markCachedDBBatchesAsUploaded(gpu);
            }
            
            
            dbIsReady = true;


            // const auto& data = fullHostDB.getData();

            // int pageableMemoryAccessUsesHostPageTables = 0;
            // int readOnlyHostRegisterSupported = 0;
            // CUDACHECK(cudaDeviceGetAttribute(&pageableMemoryAccessUsesHostPageTables, cudaDevAttrPageableMemoryAccessUsesHostPageTables, 0));
            // std::cout << "pageableMemoryAccessUsesHostPageTables " << pageableMemoryAccessUsesHostPageTables << "\n";
            // //CUDACHECK(cudaDeviceGetAttribute(&readOnlyHostRegisterSupported, cudaDeviceAttrReadOnlyHostRegisterSupported, 0));
            // // std::cout << "readOnlyHostRegisterSupported " << readOnlyHostRegisterSupported << "\n";

            // cudaDeviceProp prop;
            // CUDACHECK(cudaGetDeviceProperties(&prop, 0));

            // std::cout << "prop.pageableMemoryAccess " << prop.pageableMemoryAccess << "\n";
            // std::cout << "prop.pageableMemoryAccessUsesHostPageTables " << prop.pageableMemoryAccessUsesHostPageTables << "\n";
            // std::cout << "prop.hostRegisterReadOnlySupported " << prop.hostRegisterReadOnlySupported << "\n";
            // std::cout << "prop.hostRegisterSupported " << prop.hostRegisterSupported << "\n";


            // CUDACHECK(cudaHostRegister((void*)data.chars(), sizeof(char) * data.numChars(), cudaHostRegisterDefault));
            // CUDACHECK(cudaHostRegister((void*)data.lengths(), sizeof(SequenceLengthT) * data.numSequences(), cudaHostRegisterDefault));
            // CUDACHECK(cudaHostRegister((void*)data.offsets(), sizeof(size_t) * data.numSequences(), cudaHostRegisterDefault));
        }

        void computeTotalNumSequencePerLengthPartition(){
            nvtx::ScopedRange sr("computeTotalNumSequencePerLengthPartition", 1);
            auto lengthBoundaries = getLengthPartitionBoundaries();
            const int numLengthPartitions = getLengthPartitionBoundaries().size();

            fullHostDB_numSequencesPerLengthPartition.resize(numLengthPartitions);

            const auto& dbData = fullHostDB.getData();
            auto partitionBegin = dbData.lengths();
            for(int i = 0; i < numLengthPartitions; i++){
                //length k is in partition i if boundaries[i-1] < k <= boundaries[i]
                SequenceLengthT searchFor = lengthBoundaries[i];
                if(searchFor < std::numeric_limits<SequenceLengthT>::max()){
                    searchFor += 1;
                }
                auto partitionEnd = std::lower_bound(
                    partitionBegin, 
                    dbData.lengths() + dbData.numSequences(), 
                    searchFor
                );
                fullHostDB_numSequencesPerLengthPartition[i] = std::distance(partitionBegin, partitionEnd);
                partitionBegin = partitionEnd;
            }
        }

        void partitionDBAmongstGpus(){
            nvtx::ScopedRange sr("partitionDBAmongstGpus", 2);
            const int numGpus = deviceIds.size();
            const int numLengthPartitions = getLengthPartitionBoundaries().size();

            numSequencesPerLengthPartitionPrefixSum.clear();
            dbPartitionsByLengthPartitioning.clear();
            subPartitionsForGpus.clear();
            lengthPartitionIdsForGpus.clear();
            numSequencesPerGpu.clear();
            numSequencesPerGpuPrefixSum.clear();

            const auto& data = fullHostDB.getData();
    
            subPartitionsForGpus.resize(numGpus);
            lengthPartitionIdsForGpus.resize(numGpus);
            numSequencesPerGpu.resize(numGpus, 0);
            numSequencesPerGpuPrefixSum.resize(numGpus, 0);
    
            numSequencesPerLengthPartitionPrefixSum.resize(numLengthPartitions, 0);
            for(int i = 0; i < numLengthPartitions-1; i++){
                numSequencesPerLengthPartitionPrefixSum[i+1] = numSequencesPerLengthPartitionPrefixSum[i] + fullHostDB_numSequencesPerLengthPartition[i];
            }
    
            for(int i = 0; i < numLengthPartitions; i++){
                size_t begin = numSequencesPerLengthPartitionPrefixSum[i];
                size_t end = begin + fullHostDB_numSequencesPerLengthPartition[i];
                dbPartitionsByLengthPartitioning.emplace_back(data, begin, end);        
            }
    
            for(int lengthPartitionId = 0; lengthPartitionId < numLengthPartitions; lengthPartitionId++){
                const auto& lengthPartition = dbPartitionsByLengthPartitioning[lengthPartitionId];        
                const auto partitionedByGpu = partitionDBdata_by_numberOfChars(lengthPartition, lengthPartition.numChars() / numGpus);
        
                assert(int(partitionedByGpu.size()) <= numGpus);
                for(int gpu = 0; gpu < numGpus; gpu++){
                    if(gpu < int(partitionedByGpu.size())){
                        subPartitionsForGpus[gpu].push_back(partitionedByGpu[gpu]);
                        lengthPartitionIdsForGpus[gpu].push_back(lengthPartitionId);
                    }else{
                        //add empty partition
                        subPartitionsForGpus[gpu].push_back(DBdataView(data, 0, 0));
                        lengthPartitionIdsForGpus[gpu].push_back(0);
                    }
                }
            }
        
            for(int i = 0; i < numGpus; i++){
                for(const auto& p : subPartitionsForGpus[i]){
                    numSequencesPerGpu[i] += p.numSequences();
                }
            }
            for(int i = 0; i < numGpus-1; i++){
                numSequencesPerGpuPrefixSum[i+1] = numSequencesPerGpuPrefixSum[i] + numSequencesPerGpu[i];
            }
        
            numSequencesPerGpu_total.resize(numGpus);
            numSequencesPerGpuPrefixSum_total.resize(numGpus);
            numSequencesPerGpuPrefixSum_total[0] = 0;

        
            for(int i = 0; i < numGpus; i++){
                size_t num = numSequencesPerGpu[i];
                numSequencesPerGpu_total[i] = num;
                if(i < numGpus - 1){
                    numSequencesPerGpuPrefixSum_total[i+1] = numSequencesPerGpuPrefixSum_total[i] + num;
                }
            }

            std::vector<size_t> sequencesInPartitions(numGpus * numLengthPartitions);
            for(int gpu = 0; gpu < numGpus; gpu++){
                assert(subPartitionsForGpus[gpu].size() == numLengthPartitions);
                for(int i = 0; i < numLengthPartitions; i++){
                    sequencesInPartitions[gpu * numLengthPartitions + i] = subPartitionsForGpus[gpu][i].numSequences();
                }
            }
            hostGpuPartitionOffsets = HostGpuPartitionOffsets(numGpus, numLengthPartitions, std::move(sequencesInPartitions));
        }

        void allocateGpuWorkingSets(){
            nvtx::ScopedRange sr("allocateGpuWorkingSets", 3);
            const int numGpus = deviceIds.size();
            workingSets.clear();
            workingSets.resize(numGpus);

            if(verbose){
                std::cout << "Allocate Memory: \n";
            }
            //nvtx::push_range("ALLOC_MEM", 0);
            helpers::CpuTimer allocTimer("ALLOC_MEM");

            for(int gpu = 0; gpu < numGpus; gpu++){
                CUDACHECK(cudaSetDevice(deviceIds[gpu]));

                size_t freeMem, totalMem;
                CUDACHECK(cudaMemGetInfo(&freeMem, &totalMem));
                constexpr size_t safety = 256*1024*1024;
                size_t memlimit = std::min(freeMem, memoryConfig.maxGpuMem);
                if(memlimit > safety){
                    memlimit -= safety;
                }

                if(verbose){
                    std::cout << "gpu " << gpu << " may use " << memlimit << " bytes. ";
                }

                const bool needsPinnedStagingBuffers = numGpus > 1;

                workingSets[gpu] = std::make_unique<GpuWorkingSet>(
                    memlimit,
                    memoryConfig.maxBatchBytes,
                    memoryConfig.maxBatchSequences,
                    subPartitionsForGpus[gpu],
                    batchPlans[gpu],
                    needsPinnedStagingBuffers
                );

                if(verbose){
                    std::cout << workingSets[gpu]->getNumBatchesInCachedDB() << " out of " << batchPlans[gpu].size() << " DB batches will be cached in gpu memory\n";
                }

                //set gpu partition table
                workingSets[gpu]->setPartitionOffsets(hostGpuPartitionOffsets);

                if(shouldUseHostFuncForTransfer()){
                    //spin up the host callback thread
                    auto noop = [](void*){};
                    CUDACHECK(cudaLaunchHostFunc(
                        cudaStreamPerThread, 
                        noop, 
                        nullptr
                    ));
                }

            }    

            if(verbose){
                allocTimer.print();
            }
        }

        void allocateGpuWorkingSetsWithExistingFullGpuDB(const std::vector<std::shared_ptr<GpuDatabaseAllocationBase>>& existingFullGpuDBAllocations){
            nvtx::ScopedRange sr("allocateGpuWorkingSetsWithExistingFullGpuDB", 3);
            const int numGpus = deviceIds.size();
            workingSets.clear();
            workingSets.resize(numGpus);

            if(verbose){
                std::cout << "Allocate Memory: \n";
            }
            //nvtx::push_range("ALLOC_MEM", 0);
            helpers::CpuTimer allocTimer("ALLOC_MEM");

            for(int gpu = 0; gpu < numGpus; gpu++){
                CUDACHECK(cudaSetDevice(deviceIds[gpu]));

                size_t freeMem, totalMem;
                CUDACHECK(cudaMemGetInfo(&freeMem, &totalMem));
                constexpr size_t safety = 256*1024*1024;
                size_t memlimit = std::min(freeMem, memoryConfig.maxGpuMem);
                if(memlimit > safety){
                    memlimit -= safety;
                }

                if(verbose){
                    std::cout << "gpu " << gpu << " may use " << memlimit << " bytes. ";
                }

                const bool needsPinnedStagingBuffers = numGpus > 1;

                workingSets[gpu] = std::make_unique<GpuWorkingSet>(
                    memlimit,
                    memoryConfig.maxBatchBytes,
                    memoryConfig.maxBatchSequences,
                    subPartitionsForGpus[gpu],
                    batchPlans[gpu],
                    existingFullGpuDBAllocations[gpu],
                    needsPinnedStagingBuffers
                );

                if(verbose){
                    std::cout << workingSets[gpu]->getNumBatchesInCachedDB() << " out of " << batchPlans[gpu].size() << " DB batches will be cached in gpu memory\n";
                }

                //set gpu partition table
                workingSets[gpu]->setPartitionOffsets(hostGpuPartitionOffsets);

                if(shouldUseHostFuncForTransfer()){
                    //spin up the host callback thread
                    auto noop = [](void*){};
                    CUDACHECK(cudaLaunchHostFunc(
                        cudaStreamPerThread, 
                        noop, 
                        nullptr
                    ));
                }

            }    

            if(verbose){
                allocTimer.print();
            }
        }

        

        void createDBBatchesForGpus(){
            nvtx::ScopedRange sr("createDBBatchesForGpus", 4);
            const int numGpus = deviceIds.size();

            batchPlans.clear();
            batchPlans.resize(numGpus);
            batchPlans_cachedDB.clear();
            batchPlans_cachedDB.resize(numGpus);
    
            for(int gpu = 0; gpu < numGpus; gpu++){
                batchPlans[gpu] = computeDbCopyPlan(
                    subPartitionsForGpus[gpu],
                    lengthPartitionIdsForGpus[gpu],
                    memoryConfig.maxBatchBytes,
                    memoryConfig.maxBatchSequences
                );
                if(verbose){
                    std::cout << "Batch plan gpu " << gpu << ": " << batchPlans[gpu].size() << " batches\n";
                }
            }
        }

        void assignBatchesToGpuMem(){
            nvtx::ScopedRange sr("createDBBatchesForGpus", 5);
            const int numGpus = deviceIds.size();
            batchPlansDstInfoVec.clear();
            batchPlansDstInfoVec.resize(numGpus);
            batchPlansDstInfoVec_cachedDB.clear();
            batchPlansDstInfoVec_cachedDB.resize(numGpus);
    
            for(int gpu = 0; gpu < numGpus; gpu++){
                CUDACHECK(cudaSetDevice(deviceIds[gpu]));
                auto& ws = *workingSets[gpu];
                if(ws.getNumBatchesInCachedDB() > 0){
                    //can cache parts of db in gpu memory

                    auto plansForCachedDB = computeDbCopyPlan(
                        subPartitionsForGpus[gpu],
                        lengthPartitionIdsForGpus[gpu],
                        sizeof(char) * ws.getNumCharsInCachedDB(),
                        ws.getNumSequencesInCachedDB()
                    );
                    assert(plansForCachedDB.size() >= 1);
                    plansForCachedDB.erase(plansForCachedDB.begin() + 1, plansForCachedDB.end());
                    batchPlans_cachedDB[gpu] = plansForCachedDB;
                    // if(verbose){
                    //     std::cout << "Cached db single batch plan " << plansForCachedDB[0] << "\n";
                    // }

                    BatchDstInfo dstInfo;
                    dstInfo.isUploaded = false;
                    dstInfo.charsPtr = ws.d_cacheddb->getCharData();
                    dstInfo.lengthsPtr = ws.d_cacheddb->getLengthData();
                    dstInfo.offsetsPtr = ws.d_cacheddb->getOffsetData();
                    batchPlansDstInfoVec_cachedDB[gpu].push_back(dstInfo);
                }

                {
                    BatchDstInfo dstInfo;
                    dstInfo.isUploaded = false;
                    dstInfo.charsPtr = ws.d_cacheddb->getCharData();
                    dstInfo.lengthsPtr = ws.d_cacheddb->getLengthData();
                    dstInfo.offsetsPtr = ws.d_cacheddb->getOffsetData();

                    for(size_t i = 0; i < ws.getNumBatchesInCachedDB(); i++){
                        batchPlansDstInfoVec[gpu].push_back(dstInfo);
                        const auto& plan = batchPlans[gpu][i];
                        dstInfo.charsPtr += plan.usedBytes;
                        dstInfo.lengthsPtr += plan.usedSeq;
                        dstInfo.offsetsPtr += plan.usedSeq;
                    }

                    for(size_t i = ws.getNumBatchesInCachedDB(), buf = 0; i < batchPlans[gpu].size(); i++, buf = (buf+1)%ws.numCopyBuffers){
                        dstInfo.charsPtr = ws.d_chardata_vec[buf].data();
                        dstInfo.lengthsPtr = ws.d_lengthdata_vec[buf].data();
                        dstInfo.offsetsPtr = ws.d_offsetdata_vec[buf].data();
                        batchPlansDstInfoVec[gpu].push_back(dstInfo);
                    }
                }
            }
        }

        void markCachedDBBatchesAsUploaded(int gpu){
            auto& ws = *workingSets[gpu];
            if(ws.getNumBatchesInCachedDB() > 0){
                batchPlansDstInfoVec_cachedDB[gpu][0].isUploaded = true;
                for(size_t i = 0; i < ws.getNumBatchesInCachedDB(); i++){
                    batchPlansDstInfoVec[gpu][i].isUploaded = true;
                }
            }
        }

        void printDBDataView(const DBdataView& view) const{
            std::cout << "Sequences: " << view.numSequences() << "\n";
            std::cout << "Chars: " << view.offsets()[0] << " - " << view.offsets()[view.numSequences()] << " (" << (view.offsets()[view.numSequences()] - view.offsets()[0]) << ")"
                << " " << view.numChars() << "\n";
        }

        void printDBDataViews(const std::vector<DBdataView>& views) const {
            size_t numViews = views.size();
            for(size_t p = 0; p < numViews; p++){
                const DBdataView& view = views[p];
        
                std::cout << "View " << p << "\n";
                printDBDataView(view);
            }
        }

        

        

        std::vector<DeviceBatchCopyToPinnedPlan> computeDbCopyPlan(
            const std::vector<DBdataView>& dbPartitions,
            const std::vector<int>& lengthPartitionIds,
            size_t MAX_CHARDATA_BYTES,
            size_t MAX_SEQ
        ) const {
            std::vector<DeviceBatchCopyToPinnedPlan> result;
        
            size_t currentCopyPartition = 0;
            size_t currentCopySeqInPartition = 0;
        
            //size_t processedSequences = 0;
            while(currentCopyPartition < dbPartitions.size()){
                
                size_t usedBytes = 0;
                size_t usedSeq = 0;
        
                DeviceBatchCopyToPinnedPlan plan;
        
                while(currentCopyPartition < dbPartitions.size()){
                    if(dbPartitions[currentCopyPartition].numSequences() == 0){
                        currentCopyPartition++;
                        continue;
                    }
        
                    //figure out how many sequences to copy to pinned
                    size_t remainingBytes = MAX_CHARDATA_BYTES - usedBytes;
                    
                    auto dboffsetsBegin = dbPartitions[currentCopyPartition].offsets() + currentCopySeqInPartition;
                    auto dboffsetsEnd = dbPartitions[currentCopyPartition].offsets() + dbPartitions[currentCopyPartition].numSequences() + 1;
                    
                    auto searchFor = dbPartitions[currentCopyPartition].offsets()[currentCopySeqInPartition] + remainingBytes + 1; // +1 because remainingBytes is inclusive
                    auto it = std::lower_bound(
                        dboffsetsBegin,
                        dboffsetsEnd,
                        searchFor
                    );
        
                    size_t numToCopyByBytes = 0;
                    if(it != dboffsetsBegin){
                        numToCopyByBytes = std::distance(dboffsetsBegin, it) - 1;
                    }
                    if(numToCopyByBytes == 0 && currentCopySeqInPartition == 0){
                        std::cout << "Warning. copy buffer size too small. skipped a db portion\n";
                        break;
                    }
                    
                    size_t remainingSeq = MAX_SEQ - usedSeq;            
                    size_t numToCopyBySeq = std::min(dbPartitions[currentCopyPartition].numSequences() - currentCopySeqInPartition, remainingSeq);
                    size_t numToCopy = std::min(numToCopyByBytes,numToCopyBySeq);
        
                    if(numToCopy > 0){
                        DeviceBatchCopyToPinnedPlan::CopyRange copyRange;
                        copyRange.lengthPartitionId = lengthPartitionIds[currentCopyPartition];
                        copyRange.currentCopyPartition = currentCopyPartition;
                        copyRange.currentCopySeqInPartition = currentCopySeqInPartition;
                        copyRange.numToCopy = numToCopy;
                        plan.copyRanges.push_back(copyRange);
        
                        if(usedSeq == 0){
                            plan.h_partitionIds.push_back(lengthPartitionIds[currentCopyPartition]);
                            plan.h_numPerPartition.push_back(numToCopy);
                        }else{
                            //if is same length partition as previous copy 
                            if(plan.h_partitionIds.back() == lengthPartitionIds[currentCopyPartition]){
                                plan.h_numPerPartition.back() += numToCopy;
                            }else{
                                //new length partition
                                plan.h_partitionIds.push_back(lengthPartitionIds[currentCopyPartition]);
                                plan.h_numPerPartition.push_back(numToCopy);
                            }
                        }
                        usedBytes += (dbPartitions[currentCopyPartition].offsets()[currentCopySeqInPartition+numToCopy] 
                            - dbPartitions[currentCopyPartition].offsets()[currentCopySeqInPartition]);
                        usedSeq += numToCopy;
        
                        currentCopySeqInPartition += numToCopy;
                        if(currentCopySeqInPartition == dbPartitions[currentCopyPartition].numSequences()){
                            currentCopySeqInPartition = 0;
                            currentCopyPartition++;
                        }
                    }else{
                        break;
                    }
                }
        
                plan.usedBytes = usedBytes;
                plan.usedSeq = usedSeq;    
                
                if(usedSeq == 0 && currentCopyPartition < dbPartitions.size() && dbPartitions[currentCopyPartition].numSequences() > 0){
                    std::cout << "Warning. copy buffer size too small. skipped a db portion. stop\n";
                    break;
                }
        
                if(plan.usedSeq > 0){
                    result.push_back(plan);
                }
            }
        
            return result;
        }

        bool shouldUseHostFuncForTransfer() const{
            const int numGpus = deviceIds.size();
            return numGpus > 1;
        }


        std::vector<size_t> fullHostDB_numSequencesPerLengthPartition;
        std::vector<size_t> numSequencesPerGpu_total;
        std::vector<size_t> numSequencesPerGpuPrefixSum_total;

        //partition chars of whole DB amongst the gpus
        std::vector<size_t> numSequencesPerLengthPartitionPrefixSum;
        std::vector<DBdataView> dbPartitionsByLengthPartitioning;
        std::vector<std::vector<DBdataView>> subPartitionsForGpus;
        std::vector<std::vector<int>> lengthPartitionIdsForGpus;
        std::vector<size_t> numSequencesPerGpu;
        std::vector<size_t> numSequencesPerGpuPrefixSum;
        std::vector<std::unique_ptr<GpuWorkingSet>> workingSets;  

        std::vector<std::vector<DeviceBatchCopyToPinnedPlan>> batchPlans;
        std::vector<std::vector<BatchDstInfo>> batchPlansDstInfoVec;

        std::vector<std::vector<DeviceBatchCopyToPinnedPlan>> batchPlans_cachedDB;
        std::vector<std::vector<BatchDstInfo>> batchPlansDstInfoVec_cachedDB;

        bool dbIsReady{};
        AnyDBWrapper fullHostDB;

        mutable std::unique_ptr<SequenceLengthStatistics> dbSequenceLengthStatistics;

        HostGpuPartitionOffsets hostGpuPartitionOffsets;

        //--------------------------------------
        bool verbose = false;

        MemoryConfig memoryConfig;
        
        std::vector<int> deviceIds;

    };


LIBMARV_NAMESPACE_WITH_NESTING_END

#endif


