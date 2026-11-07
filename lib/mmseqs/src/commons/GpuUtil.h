#ifndef GPUUTIL_H
#define GPUUTIL_H

#include <atomic>
#include <string>
#include <cstddef>
#include "marv.h"

struct GPUSharedMemory {
    enum State {
        IDLE,
        RESERVED,
        READY,
        DONE
    };

    unsigned int maxSeqLen;                   // Maximum length of the sequence
    unsigned int maxResListLen;               // Maximum length of the results list
    std::atomic<int>  state{IDLE};            // State of the shared memory
    std::atomic<bool> serverExit{false};      // Has server exited
    unsigned int queryOffset;                 // Offset to the query data
    unsigned int queryLen;                    // Length of the query sequence
    unsigned int resultsOffset;               // Offset to the results data
    unsigned int resultLen;                   // Length of the result list
    unsigned int profileOffset;               // Offset to the profile data
    unsigned int profile12StOffset;           // Offset to the 12-state profile data (packed 3Di+12-state db)
    bool packedDb;                            // Server db is packed 3Di+12-state
    int topNSize;                             // Requested result list length (packed db)
    int maskThreshold;                        // 3Di repeat masking threshold (packed db)
    char maskLetter;                          // 3Di masking letter (packed db)

    static const unsigned int ROWS_12ST = 12;

    // Get pointers to the query, results, and profile data sections
    int8_t* getQueryPtr() { return reinterpret_cast<int8_t*>(this) + queryOffset; }
    Marv::Result* getResultsPtr() { return reinterpret_cast<Marv::Result*>(reinterpret_cast<char*>(this) + resultsOffset); }
    int8_t* getProfilePtr() { return reinterpret_cast<int8_t*>(this) + profileOffset; }
    int8_t* getProfile12StPtr() { return reinterpret_cast<int8_t*>(this) + profile12StOffset; }

    // Calculate the total size needed for the shared memory
    static size_t calculateSize(unsigned int maxSeqLen, unsigned int maxResListLen) {
        return sizeof(GPUSharedMemory) +
               sizeof(char) * maxSeqLen +              // Size for query data
               sizeof(Marv::Result) * maxResListLen +  // Size for results data
               sizeof(int8_t) * 21 * maxSeqLen +       // Size for profile data
               sizeof(int8_t) * ROWS_12ST * maxSeqLen; // Size for 12-state profile data
    }

    static std::string getShmHash(const std::string& db);

    // Allocate and initialize shared memory
    static GPUSharedMemory* alloc(const std::string& name, unsigned int maxSeqLen, unsigned int maxResListLen);

    // Deallocate shared memory
    static void dealloc(GPUSharedMemory* layout, const std::string& name);

    static void unmap(GPUSharedMemory* layout);

    // Function to open and map existing shared memory and automatically determine sizes
    static GPUSharedMemory* openSharedMemory(const std::string& name);

};

#endif
