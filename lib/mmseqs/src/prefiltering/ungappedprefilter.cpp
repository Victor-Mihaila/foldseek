//
// Created by Martin Steinegger on 17.09.18.
//

#include "DistanceCalculator.h"
#include "Util.h"
#include "Parameters.h"
#include "Matcher.h"
#include "Debug.h"
#include "DBReader.h"
#include "DBWriter.h"
#include "QueryMatcher.h"
#include "NucleotideMatrix.h"
#include "FastSort.h"
#include "SubstitutionMatrixProfileStates.h"
#include "IndexReader.h"
#include "QueryMatcherTaxonomyHook.h"
#include "Masker.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <chrono>
#include <thread>

#ifdef OPENMP
#include <omp.h>
#endif
// #define HAVE_CUDA 1
#ifdef HAVE_CUDA
#include "GpuUtil.h"
#include "Alignment.h"
#include <signal.h>
#endif

#ifdef HAVE_CUDA

volatile sig_atomic_t keepRunningClient = 1;
void intHandlerClient(int) {
    keepRunningClient = 0;
}

void runFilterOnGpu(Parameters & par, BaseMatrix * subMat, BaseMatrix * subMat12St,
                    DBReader<unsigned int> * qdbr, DBReader<unsigned int> * tdbr,
                    bool sameDB, DBWriter & resultWriter, EvalueComputation * evaluer,
                    QueryMatcherTaxonomyHook *taxonomyHook){
    Debug::Progress progress(qdbr->getSize());
    const int querySeqType = qdbr->getDbtype();
    Sequence qSeq(par.maxSeqLen, querySeqType, subMat, 0, false, par.compBiasCorrection);

    // ---- Foldseek GPU 12-state (3Di + 12st) two-pass scan ----
    // subMat12St is non-NULL only when the target GPU DB carries the packed 3Di+12st byte
    // (gated in prefilterInternal). The packed DB stores `state3Di * 12 + state12st` raw and
    // unmasked; libmarv splits it on the fly (byte/12, byte%12), masks the 3Di channel with
    // mask-n-repeat, scores 3Di gaplessly over all targets, keeps the top N, and adds the 12st
    // diagonal score for those (Marv::search_scoreOnly_foldseek).
    //
    // The query side is ours: mask the query, then fold the mask into both query profiles.
    // The foldseek entry point takes *only* profiles (no query sequence), so anything the
    // kernel should know about the query has to be expressed in the profile.
    //
    // A NULL subMat12St keeps the legacy single-channel 3Di path (Marv::search_scoreOnly).
    //
    // Two separate questions, and they must not be conflated:
    //
    //   packedTarget -- is the *target* DB packed? If so we MUST use the foldseek entry point,
    //     because it is the only one that splits the byte (/12) before scoring. The plain
    //     score-only kernels apply ClampToInvalid (>20 -> 20) straight to the stored byte, and
    //     packed bytes run to 251, so every target residue would collapse to 'X'. That silently
    //     produced garbage for profile queries (iterative search) before this split existed.
    //
    //   have12StQuery -- do we have a real 12-state *query* profile to contribute? Only when the
    //     query DB carries the aux channel. Profile queries (--num-iterations >= 2) do not: the
    //     12st profile lives in a separate `_ss12` DB that this command is never handed. When we
    //     don't have one we still take the foldseek path, but with an all-zero 12st PSSM, which
    //     contributes exactly 0 (diagonal_rescore starts max12st_score at 0 and only max'es it).
    //     Net effect: correct 3Di-only scoring on a packed DB instead of an all-'X' target.
    const Sequence::SeqAuxInfo *auxInfo = Sequence::getAuxInfo(querySeqType);
    // The 12st query profile must have exactly Alphabet12St::STATE_CNT (12) rows: libmarv views
    // it as a 12 x queryLength mdspan and indexes rows by the target's 12st state (byte % 12).
    // subMat12St->alphabetSize is 21 (mat12st.out is a padded 21-letter matrix), so it must not
    // be used as the row count here. Fall back to the hard-coded 12 when the query DB has no aux
    // channel to describe it, since the buffer still has to match what the kernel reads.
    const int rows12St = (auxInfo != NULL && auxInfo->auxAlphabetSize > 0)
        ? (int)auxInfo->auxAlphabetSize : 12;
    // Read packedness off the target DB, NOT off subMat12St. The two are independent: --aux-score
    // (--ss-12st) can switch the 12st *scoring* off while the DB is still packed, and a packed DB
    // must go through the foldseek entry point regardless, because it is the only one that splits
    // the byte before scoring. Inferring packedness from subMat12St would send a packed DB down
    // the clamping path and turn every target residue into 'X'.
    const uint16_t EXT_3DI_12ST_Q = 32;
    const bool packedTarget =
        (DBReader<unsigned int>::getExtendedDbtype(tdbr->getDbtype()) & EXT_3DI_12ST_Q) != 0;
    const bool have12StQuery =
        packedTarget && (subMat12St != NULL) && (qSeq.numSequenceAux != NULL);
    if (packedTarget && have12StQuery == false) {
        Debug(Debug::WARNING) << "GPU prefilter scores 3Di only for this run "
                              << "(12-state channel disabled, or query DB has none)\n";
    }
    // libmarv rescores exactly the top N of the 3Di pass, so the 12st channel can only reorder
    // within that set. Ask for a multiple of --max-seqs and truncate to --max-seqs on the CPU
    // after rescoring, so 12st can promote hits that 3Di alone ranked below the cutoff.
    // With a zero 12st profile there is nothing to reorder, so don't inflate the pool.
    const size_t topN = have12StQuery
        ? std::min(tdbr->getSize(), (size_t)std::max(1, par.gpuRescoreTopkMult) * par.maxResListLen)
        : par.maxResListLen;

    const bool isProfileQuery = Parameters::isEqualDbtype(querySeqType, Parameters::DBTYPE_HMM_PROFILE);

    // ---- query batching ----
    // libmarv builds its aligner objects and a ~10M-element working set *inside every* search_*
    // call. Calling the single-query overload once per query therefore pays that setup per query:
    // measured 6.6 ms/query, against 0.38 ms/query for the old persistent-CudaSW4 API -- a 17x
    // regression in the prefilter stage. The batched overloads amortise it over a whole batch.
    //
    // Server mode keeps a batch of 1: its shared-memory protocol exchanges one query at a time.
    //
    // Batch size is bounded by libmarv, not by us. Its working set does
    //     d_kernelOutputArrayScores.resize(kernelOutputArraySize * maxNumQueries)
    // (searches/score_only_search.cuh) with both operands `int`, and marv.cu hardcodes
    // kernelOutputArraySize = 10'000'000. So with unpatched libmarv:
    //   - beyond 214 queries that product overflows int32 and the allocation explodes
    //     (256 asks cudaMalloc for ~1.8e19 bytes and aborts on std::bad_alloc), and
    //   - each query in the batch costs ~80 MB of GPU working set (4 B score + 4 B index
    //     per output slot), so the batch is memory-bound well before that.
    //
    // 32 is deliberately conservative: 32 * 10'000'000 stays inside int32, and the working set
    // is ~2.5 GB unpatched / ~3 MB if kernelOutputArraySize is ever sized to the DB. Measured on
    // SCOPe (11k queries, 3Di-only prefilter stage): per-query 73.7 s -> batch 32 48.1 s with the
    // 10M array; 17.3 s -> 8.8 s with a DB-sized array. Going to 256 only bought 0.3 s, so there
    // is nothing to gain from a larger batch here.
    const size_t batchMax = (par.gpuServer == 0) ? 32 : 1;
    // Per-slot buffers. Each grows to the longest query that lands in that slot, so the footprint
    // follows the DB's length distribution rather than par.maxSeqLen (which is 65535 by default).
    std::vector<std::vector<int8_t> > batchProfileBuf(batchMax);
    std::vector<std::vector<int8_t> > batchProfile12StBuf(batchMax);
    std::vector<std::vector<unsigned char> > batchSeqBuf(batchMax);
    // What libmarv is handed, plus what we need to reconstruct per-query context afterwards.
    std::vector<int> batchLen;
    std::vector<int8_t*> batchPssm;
    std::vector<int8_t*> batchPssm12St;
    std::vector<const char*> batchSeqPtr;
    std::vector<unsigned int> batchKey;
    batchLen.reserve(batchMax);
    batchPssm.reserve(batchMax);
    batchPssm12St.reserve(batchMax);
    batchSeqPtr.reserve(batchMax);
    batchKey.reserve(batchMax);

    std::vector<Marv::Result> results;
    // NOTE: reserve (not resize) because Marv::Result has no default constructor; libmarv writes
    // through results.data() and reports the count in Stats::results. The batched overloads write
    // query b's block at results.data() + b * topN.
    results.reserve(batchMax * topN);
    std::vector<hit_t> shortResults;
    std::vector<Matcher::result_t> resultsAln;
    // Masker over the 3Di matrix; used for mask-n-repeat on the query. tantan is disabled below.
    // Only meaningful when we have both channels to keep in sync; a profile query's PSSM is
    // prebuilt and is left alone.
    Masker *queryMasker = have12StQuery ? new Masker(*subMat) : NULL;
    const unsigned char x12St = have12StQuery ? subMat12St->aa2num[static_cast<int>('X')] : 0;
    // 3Di 'X' is what libmarv writes over masked target positions; keep the query consistent.
    const unsigned char x3Di = subMat->aa2num[static_cast<int>('X')];
    // libmarv masks target runs of length >= maskingThreshold. mmseqs uses --mask-n-repeat == 0
    // to mean "no repeat masking", which would degenerate to "mask every position" there, so map
    // it (and anything out of range) onto a threshold no real run reaches. libmarv's masking
    // kernel additionally requires 2 * threshold < 1024.
    const int MARV_MAX_MASK_THRESHOLD = 511;
    int maskThreshold = (par.maskNrepeats > 0) ? par.maskNrepeats : MARV_MAX_MASK_THRESHOLD;
    if (maskThreshold > MARV_MAX_MASK_THRESHOLD) {
        Debug(Debug::WARNING) << "--mask-n-repeat " << maskThreshold << " exceeds the GPU limit; "
                              << "clamping to " << MARV_MAX_MASK_THRESHOLD << "\n";
        maskThreshold = MARV_MAX_MASK_THRESHOLD;
    }

    std::string resultBuffer;
    resultBuffer.reserve(262144);
    char buffer[1024+32768];

    size_t compBufferSize = (par.maxSeqLen + 1) * sizeof(float);
    float *compositionBias = NULL;
    if (par.compBiasCorrection == true) {
        compositionBias = (float*)malloc(compBufferSize);
        memset(compositionBias, 0, compBufferSize);
    }

    std::string hash = "";
    if (par.gpuServer != 0) {
        hash = GPUSharedMemory::getShmHash(par.db2);
        std::string path = "/dev/shm/" + hash;
        // Debug(Debug::WARNING) << path << "\n";
        int waitTimeout = par.gpuServerWaitTimeout;
        std::chrono::steady_clock::time_point startTime = std::chrono::steady_clock::now();
        bool statusPrinted = false;
        while (true) {
            size_t shmSize = FileUtil::getFileSize(path);
            // server is ready once the shm file exists and is not 0 byte large
            if (shmSize != (size_t)-1 && shmSize > 0) {
                break;
            }

            if (waitTimeout == 0) {
                Debug(Debug::ERROR) 
                    << "gpuserver for database " << par.db2 << " not found.\n"
                    << "Please start gpuserver with the same CUDA_VISIBLE_DEVICES\n";
                EXIT(EXIT_FAILURE);
            }

            if (waitTimeout > 0) {
                if (statusPrinted == false) {
                    Debug(Debug::WARNING) << "Waiting for `gpuserver`\n";
                    statusPrinted = true;
                } else {
                    Debug(Debug::WARNING) << ".";
                }
                std::chrono::steady_clock::time_point now = std::chrono::steady_clock::now();
                auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - startTime).count();
                if (elapsed >= waitTimeout) {
                    Debug(Debug::ERROR)
                        << "\ngpuserver for database " << par.db2 << " not found after " << elapsed <<  "seconds.\n"
                        << "Please start gpuserver with the same CUDA_VISIBLE_DEVICES\n";
                    EXIT(EXIT_FAILURE);
                }
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
        }
        if (waitTimeout > 0 && statusPrinted) {
            Debug(Debug::INFO) << "\n";
        }
    }

    size_t* offsetData = NULL;
    int32_t* lengthData = NULL;
    std::vector<size_t> offsets;
    std::vector<int32_t> lengths;
    GPUSharedMemory* layout = NULL;
    if (hash.empty()) {
        offsets.reserve(tdbr->getSize() + 1);
        lengths.reserve(tdbr->getSize());
        for (size_t id = 0; id < tdbr->getSize(); id++) {
            offsets.emplace_back(tdbr->getIndex()[id].offset);
            lengths.emplace_back(tdbr->getIndex()[id].length - 2);
        }
        offsets.emplace_back(offsets.back() + lengths.back());
        offsetData = offsets.data();
        lengthData = lengths.data();
    } else {
        layout = GPUSharedMemory::openSharedMemory(hash);
    }

    const bool serverMode = par.gpuServer;
    Marv* marv = NULL;
    if (serverMode == 0) {
       if (offsetData == NULL || lengthData == NULL) {
           Debug(Debug::ERROR) << "Invalid GPU database\n";
           EXIT(EXIT_FAILURE);
       }
        marv = new Marv(tdbr->getSize(), subMat->alphabetSize);
        void* h = marv->loadDb(
            tdbr->getDataForFile(0), offsetData, lengthData, tdbr->getDataSizeForFile(0)
        );
        marv->setDb(h);
    } else if (layout == NULL) {
       Debug(Debug::ERROR) << "No GPU server shared memory connection\n";
       EXIT(EXIT_FAILURE);
    } else {
        struct sigaction act;
        // Set up the handler for SIGINT and SIGTERM
        memset(&act, 0, sizeof(act));
        act.sa_handler = intHandlerClient;
        sigaction(SIGINT, &act, NULL);
        sigaction(SIGTERM, &act, NULL);
    }

    // marv.prefetch();
    for (size_t batchStart = 0; batchStart < qdbr->getSize(); batchStart += batchMax) {
        if (!keepRunningClient) {
            break;
        }
        batchLen.clear();
        batchPssm.clear();
        batchPssm12St.clear();
        batchSeqPtr.clear();
        batchKey.clear();

        // ---- phase 1: build the query profiles for this batch ----
        const size_t batchEnd = std::min(batchStart + batchMax, qdbr->getSize());
        for (size_t id = batchStart; id < batchEnd; id++) {
        const size_t slot = id - batchStart;
        size_t queryKey = qdbr->getDbKey(id);
        unsigned int querySeqLen = qdbr->getSeqLen(id);
        char *querySeqData = qdbr->getData(id, 0);
        qSeq.mapSequence(id, queryKey, querySeqData, querySeqLen);
        int8_t* profile = NULL;
        int8_t* profile12St = NULL;
        // The 12st buffer is sized for every query regardless of which branch fills it, so the
        // zero-PSSM case (profile queries, or a query DB without the aux channel) always hands
        // libmarv a correctly sized, fully zeroed 12 x L block.
        if (packedTarget) {
            batchProfile12StBuf[slot].assign((size_t)rows12St * (size_t)qSeq.L, 0);
            profile12St = batchProfile12StBuf[slot].data();
        }
        if (isProfileQuery) {
            // profile_for_alignment is owned by the Sequence and is overwritten by the next
            // mapSequence(), so it has to be copied into the slot rather than aliased.
            batchProfileBuf[slot].assign(qSeq.profile_for_alignment,
                                         qSeq.profile_for_alignment + (size_t)subMat->alphabetSize * (size_t)qSeq.L);
            profile = batchProfileBuf[slot].data();
        } else {
            if (have12StQuery) {
                // mask-n-repeat on the query (tantan disabled by design on this path); mirror the
                // mask onto the 12st channel so both query profiles agree on masked positions.
                // libmarv masks the targets on the fly, so only the query is masked here.
                //
                // maskLowerCaseLetter MUST stay false here. It tests islower() on the raw DB byte,
                // and a packed 3Di+12st DB uses the whole 0..251 range with no lowercase soft-mask
                // convention -- so on SCOPe ~9% of residues have a packed byte that lands in
                // 'a'..'z' and would be masked to X for no reason (vs ~3% legitimately masked by
                // mask-n-repeat). par.maskLowerCaseMode is 1 by default in foldseek, and a packed
                // _ss DB reports base dbtype AMINO_ACIDS, so the guard inside Masker does not
                // save us.
                queryMasker->maskSequence(qSeq, /*maskTantan*/ false, par.maskProb,
                                          /*maskLowerCaseLetter*/ false, par.maskNrepeats);
                for (int i = 0; i < qSeq.L; ++i) {
                    if (qSeq.numSequence[i] == (unsigned char)queryMasker->maskLetterNum) {
                        qSeq.numSequenceAux[i] = x12St;
                    }
                }
            }
            batchProfileBuf[slot].resize((size_t)subMat->alphabetSize * (size_t)qSeq.L);
            profile = batchProfileBuf[slot].data();
            if (compositionBias != NULL) {
                if ((size_t)qSeq.L >= compBufferSize) {
                    compBufferSize = (size_t)qSeq.L * 1.5 * sizeof(float);
                    compositionBias = (float*)realloc(compositionBias, compBufferSize);
                    // memset(compositionBias, 0, compBufferSize);
                }
                SubstitutionMatrix::calcLocalAaBiasCorrection(subMat, qSeq.numSequence, qSeq.L, compositionBias, par.compBiasCorrectionScale);
            }
            for (size_t j = 0; j < (size_t)subMat->alphabetSize; ++j) {
                for (size_t i = 0; i < (size_t)qSeq.L; ++i) {
                    short bias = 0;
                    if (compositionBias != NULL) {
                        bias = static_cast<short>((compositionBias[i] < 0.0) ? (compositionBias[i] - 0.5) : (compositionBias[i] + 0.5));
                    }
                    profile[j * qSeq.L  + i] = subMat->subMatrix[j][qSeq.numSequence[i]] + bias;
                }
            }
            // Second query profile over the 12st channel: exactly rows12St (=12) rows, indexed by
            // the target's 12st state, columns = query positions. No composition-bias term (the
            // 3Di channel already carries one); add one here later if it helps.
            if (have12StQuery) {
                for (size_t j = 0; j < (size_t)rows12St; ++j) {
                    for (size_t i = 0; i < (size_t)qSeq.L; ++i) {
                        profile12St[j * qSeq.L + i] = subMat12St->subMatrix[j][qSeq.numSequenceAux[i]];
                    }
                }
            }
        }
        // The non-foldseek entry points also take the encoded query sequence, and qSeq is reused
        // by the next iteration, so copy it into the slot as well.
        batchSeqBuf[slot].assign(qSeq.numSequence, qSeq.numSequence + qSeq.L);
        batchSeqPtr.emplace_back(reinterpret_cast<const char*>(batchSeqBuf[slot].data()));
        batchLen.emplace_back(qSeq.L);
        batchPssm.emplace_back(profile);
        batchPssm12St.emplace_back(profile12St);
        batchKey.emplace_back(queryKey);
        }   // end phase 1

        // ---- phase 2: one libmarv call for the whole batch ----
        const size_t batchCount = batchLen.size();
        std::vector<Marv::Stats> statsVec;
        if (serverMode == 0) {
            if (packedTarget) {
                // 3Di gapless over all targets -> top N -> add the 12st diagonal score for those.
                // Targets are split and masked inside the kernel; maskingLetter3Di must be the
                // same 'X' the query profile was masked with. No gap penalties (gapless only).
                // Note: this entry point reports no end positions (they stay -1), which is fine
                // because --gpu forces PREF_MODE_UNGAPPED, where end positions are unused.
                //
                // Taken for ANY packed target, even when profile12St is all zeros: it is the only
                // entry point that splits the packed byte, so it is also how a 3Di-only score is
                // obtained correctly from a packed DB.
                statsVec = marv->search_scoreOnly_foldseek(
                    results.data(), batchLen, batchPssm, batchPssm12St, (int)topN,
                    /*int8IsAllowed*/ true, /*handleOverflows*/ true,
                    /*maskingLetter3di*/ (char)x3Di, /*maskingThreshold*/ maskThreshold);
            } else if (par.prefMode == Parameters::PREF_MODE_UNGAPPED_AND_GAPPED) {
                // gapless scan over all targets, then Smith-Waterman with end positions for the
                // tops. gop/gex match the values libmarv used before it took them per call.
                statsVec = marv->search_scoreOnly_then_scoreEndpos_for_tops(
                    results.data(), batchSeqPtr, batchLen, batchPssm,
                    (int)topN, /*gop*/ -11, /*gex*/ -1,
                    Marv::AlignmentType::GAPLESS, /*int8IsAllowed*/ true, /*handleOverflows*/ true,
                    Marv::AlignmentType::SMITH_WATERMAN_ENDPOS, /*int8IsAllowed*/ false, /*handleOverflows*/ true);
            } else {
                statsVec = marv->search_scoreOnly(
                    results.data(), batchSeqPtr, batchLen, batchPssm,
                    (int)topN, Marv::AlignmentType::GAPLESS, /*gop*/ -11, /*gex*/ -1,
                    /*int8IsAllowed*/ true, /*handleOverflows*/ true);
            }
        } else {
            // NOTE: GPU-server mode does not carry the 12st channel. The shared-memory layout
            // below ships only the 3Di query + profile, so a packed DB run through the server
            // falls back to single-channel 3Di ranking. Extending GPUSharedMemory with the 12st
            // profile + a server-side foldseek scan is a follow-up.
            // batchMax is 1 here, so this handles exactly one query per iteration.
            statsVec.resize(1);
            const int qLen = batchLen[0];
            const unsigned char* qNum = batchSeqBuf[0].data();
            const int8_t* qPssm = batchPssm[0];
            bool claimed = false;
            while (!claimed) {
                if (layout->serverExit.load(std::memory_order_acquire) == true) {
                    // server has shut down
                    Debug(Debug::ERROR) << "GPU server has unexpectedly shut down\n";
                    EXIT(EXIT_FAILURE);
                }
                if (keepRunningClient == false) {
                    EXIT(EXIT_FAILURE);
                }

                int expected = GPUSharedMemory::IDLE;
                int desired = GPUSharedMemory::RESERVED;
                if (layout->state.compare_exchange_strong(expected, desired, std::memory_order_acq_rel)) {
                    // Debug(Debug::ERROR) << "switch to reserved\n";
                    claimed = true;
                    memcpy(layout->getQueryPtr(), qNum, qLen);
                    memcpy(layout->getProfilePtr(), qPssm, subMat->alphabetSize * qLen);
                    layout->queryLen = qLen;
                    std::atomic_thread_fence(std::memory_order_release);
                    // Debug(Debug::ERROR) << "switch to ready\n";
                    layout->state.store(GPUSharedMemory::READY, std::memory_order_release);

                    while (true) {
                        if (layout->serverExit.load(std::memory_order_acquire) == true) {
                            Debug(Debug::ERROR) << "GPU server has unexpectedly shut down\n";
                            EXIT(EXIT_FAILURE);
                        }

                        if (layout->state.load(std::memory_order_acquire) == GPUSharedMemory::DONE) {
                            break;
                        } else {
                            std::this_thread::yield();
                        }
                    }

                    std::atomic_thread_fence(std::memory_order_acquire);
                    memcpy(results.data(), layout->getResultsPtr(), layout->resultLen * sizeof(Marv::Result));
                    statsVec[0].results = layout->resultLen;
                    // Debug(Debug::ERROR) << "switch to idle\n";
                    layout->state.store(GPUSharedMemory::IDLE, std::memory_order_release);
                    if (keepRunningClient == false) {
                        EXIT(EXIT_FAILURE);
                    }
                } else {
                    std::this_thread::yield();
                }
            }
        }
        if (keepRunningClient == false) {
            EXIT(EXIT_FAILURE);
        }

        // ---- phase 3: replay the batch, one query at a time ----
        for (size_t slot = 0; slot < batchCount; slot++) {
        const unsigned int queryKey = batchKey[slot];
        const int queryLen = batchLen[slot];
        // libmarv writes query `slot`'s hits at results.data() + slot * topN.
        const Marv::Result* slotResults = results.data() + slot * topN;
        const size_t slotResultCount = statsVec[slot].results;

        for(size_t i = 0; i < slotResultCount; i++){
            unsigned int targetKey = tdbr->getDbKey(slotResults[i].id);
            int score = slotResults[i].score;
            if(taxonomyHook != NULL){
                TaxID currTax = taxonomyHook->taxonomyMapping->lookup(targetKey);
                if (taxonomyHook->expression[0]->isAncestor(currTax) == false) {
                    continue;
                }
            }
            // check if evalThr != inf
            // double evalue = 0.0;
            // if (par.evalThr < std::numeric_limits<double>::max()) {
            //     evalue = evaluer->computeEvalue(score, qSeq.L);
            // }
            // bool hasEvalue = (evalue <= par.evalThr);
            bool hasDiagScore = (score > par.minDiagScoreThr);

            const bool isIdentity = (queryKey == targetKey && (par.includeIdentity || sameDB))? true : false;
            // --filter-hits
            if (isIdentity || hasDiagScore) {
                if(par.prefMode == Parameters::PREF_MODE_UNGAPPED_AND_GAPPED){
                    Matcher::result_t res;
                    res.dbKey = targetKey;
                    res.eval = evaluer->computeEvalue(score, queryLen);
                    res.dbEndPos = slotResults[i].dbEndPos;
                    res.dbLen = tdbr->getSeqLen(slotResults[i].id);
                    res.qEndPos =  slotResults[i].qEndPos;
                    res.qLen = queryLen;
                    unsigned int qAlnLen = std::max(static_cast<unsigned int>(res.qEndPos), static_cast<unsigned int>(1));
                    unsigned int dbAlnLen = std::max(static_cast<unsigned int>(res.dbEndPos), static_cast<unsigned int>(1));
                    //seqId = (alignment.score1 / static_cast<float>(std::max(dbAlnLen, qAlnLen)))  * 0.1656 + 0.1141;
                    res.seqId = Matcher::estimateSeqIdByScorePerCol(score, qAlnLen, dbAlnLen);
                    res.qcov = SmithWaterman::computeCov(0, res.qEndPos, res.qLen );
                    res.dbcov = SmithWaterman::computeCov(0, res.dbEndPos, res.dbLen );
                    res.score = evaluer->computeBitScore(score);
                    if(Alignment::checkCriteria(res, isIdentity, par.evalThr,  par.seqIdThr,  par.alnLenThr,  par.covMode,  par.covThr)){
                        resultsAln.emplace_back(res);
                    }
                } else {
                    hit_t hit;
                    hit.seqId = targetKey;
                    hit.prefScore = score;
                    hit.diagonal = 0;
                    shortResults.emplace_back(hit);
                }
            }
        }
        if(par.prefMode == Parameters::PREF_MODE_UNGAPPED_AND_GAPPED) {
            SORT_PARALLEL(resultsAln.begin(), resultsAln.end(), Matcher::compareHits);
            size_t maxSeqs = std::min(par.maxResListLen, resultsAln.size());
            for (size_t i = 0; i < maxSeqs; ++i) {
                size_t len = Matcher::resultToBuffer(buffer, resultsAln[i], false);
                resultBuffer.append(buffer, len);
            }
        }else{
            SORT_PARALLEL(shortResults.begin(), shortResults.end(), hit_t::compareHitsByScoreAndId);
            size_t maxSeqs = std::min(par.maxResListLen, shortResults.size());
            for (size_t i = 0; i < maxSeqs; ++i) {
                size_t len = QueryMatcher::prefilterHitToBuffer(buffer, shortResults[i]);
                resultBuffer.append(buffer, len);
            }
        }

        resultWriter.writeData(resultBuffer.c_str(), resultBuffer.length(), queryKey, 0);
        resultBuffer.clear();
        shortResults.clear();
        resultsAln.clear();
        progress.updateProgress();
        }   // end phase 3
    }
    if (marv != NULL) {
        delete marv;
    } else {
        GPUSharedMemory::unmap(layout);
    }

    if (compositionBias != NULL) {
        free(compositionBias);
    }
    // batchProfileBuf / batchProfile12StBuf / batchSeqBuf are std::vectors and free themselves
    if (queryMasker != NULL) {
        delete queryMasker;
    }
}
#endif

// CPU mirror of the GPU 12-state rescore. libmarv's diagonal_rescore_simple_kernel
// (diagonal_rescore.cuh) takes the diagonal of the 3Di alignment,
//
//     bestDiagonal = queryEndExcl - subjectEndExcl        (query on x, subject on y)
//
// walks that whole diagonal accumulating a relu running sum of 12st substitution scores
// (add_relu(a,b) = max(a+b,0)) and adds its maximum to the 3Di score. max12st_score starts at 0,
// so the contribution is always >= 0.
//
// The diagonal comes from SmithWaterman::ungapped_alignment's tracking overload, which reports
// where its maximum was, so nothing has to be rescanned here: this is O(diagonal length), not
// O(qLen * tLen).
static int score12StOnDiagonal(const unsigned char *q12St, int qLen,
                               const unsigned char *t12St, int tLen,
                               BaseMatrix *subMat12St, int diagonal) {
    if (qLen <= 0 || tLen <= 0) {
        return 0;
    }
    int q = (diagonal > 0) ? diagonal : 0;
    int s = (diagonal < 0) ? -diagonal : 0;
    if (q >= qLen || s >= tLen) {
        return 0;
    }
    const int cells = std::min(qLen - q, tLen - s);
    int segScore = 0;
    int max12St = 0;
    for (int k = 0; k < cells; k++) {
        segScore += subMat12St->subMatrix[t12St[s + k]][q12St[q + k]];
        if (segScore < 0) {
            segScore = 0;
        }
        if (segScore > max12St) {
            max12St = segScore;
        }
    }
    return max12St;
}

void runFilterOnCpu(Parameters & par, BaseMatrix * subMat, BaseMatrix * subMat12St, int8_t * tinySubMat,
                    DBReader<unsigned int> * qdbr, DBReader<unsigned int> * tdbr,
                    SequenceLookup * sequenceLookup, bool sameDB, DBWriter & resultWriter, EvalueComputation * evaluer,
                    QueryMatcherTaxonomyHook *taxonomyHook, int alignmentMode){
    std::vector<hit_t> shortResults;
    shortResults.reserve(tdbr->getSize()/2);
    Debug::Progress progress(qdbr->getSize());
    const int targetSeqType = tdbr->getDbtype();
    const int querySeqType = qdbr->getDbtype();
    // Same workflow the GPU path uses: score 3Di over every target, keep the top
    // (--gpu-rescore-topk-mult * --max-seqs) by 3Di, add the 12st channel to those, re-rank, then
    // truncate to --max-seqs. Set only when the target DB is packed and --aux-score is on; the
    // *query* must carry the aux channel too, which profile queries do not (mirrors
    // runFilterOnGpu's have12StQuery). rescoreCount is shared across the team.
    const bool queryHasAux = (Sequence::getAuxInfo(querySeqType) != NULL)
                             && (Sequence::getAuxInfo(querySeqType)->auxRemap != NULL);
    const bool targetHasAux = (Sequence::getAuxInfo(targetSeqType) != NULL)
                              && (Sequence::getAuxInfo(targetSeqType)->auxRemap != NULL);
    const bool use12St = (subMat12St != NULL) && queryHasAux && targetHasAux
                         && (alignmentMode == 0)
                         && (Parameters::isEqualDbtype(querySeqType, Parameters::DBTYPE_HMM_PROFILE) == false);
    if (subMat12St != NULL && use12St == false) {
        Debug(Debug::INFO) << "12-state channel unavailable on the CPU ungapped prefilter "
                           << "(needs a non-profile query with the aux channel and --alignment-mode 0); "
                           << "scoring 3Di only\n";
    }
    const size_t topN12St = use12St
        ? std::max((size_t)1, (size_t)par.gpuRescoreTopkMult * par.maxResListLen)
        : 0;
    size_t rescoreCount = 0;
#ifdef OPENMP
    omp_set_nested(1);
#endif

#pragma omp parallel
    {
        unsigned int thread_idx = 0;
#ifdef OPENMP
        thread_idx = (unsigned int) omp_get_thread_num();
#endif
        char buffer[1024+32768];
        std::vector<hit_t> threadShortResults;
        Sequence qSeq(par.maxSeqLen, querySeqType, subMat, 0, false, par.compBiasCorrection);
        Sequence tSeq(par.maxSeqLen, targetSeqType, subMat, 0, false, par.compBiasCorrection);
        SmithWaterman aligner(par.maxSeqLen, subMat->alphabetSize,
                              par.compBiasCorrection, par.compBiasCorrectionScale, NULL);

        std::string resultBuffer;
        resultBuffer.reserve(262144);
        for (size_t id = 0; id < qdbr->getSize(); id++) {
            char *querySeqData = qdbr->getData(id, thread_idx);
            size_t queryKey = qdbr->getDbKey(id);
            unsigned int querySeqLen = qdbr->getSeqLen(id);

            qSeq.mapSequence(id, queryKey, querySeqData, querySeqLen);
//            qSeq.printProfileStatePSSM();
            if(Parameters::isEqualDbtype(qSeq.getSeqType(), Parameters::DBTYPE_HMM_PROFILE) ){
                aligner.ssw_init(&qSeq, qSeq.getAlignmentProfile(), subMat);
            }else{
                aligner.ssw_init(&qSeq, tinySubMat, subMat);
            }
#pragma omp for schedule(static) nowait
            for (size_t tId = 0; tId < tdbr->getSize(); tId++) {
                unsigned int targetKey = tdbr->getDbKey(tId);
                if(taxonomyHook != NULL){
                    TaxID currTax = taxonomyHook->taxonomyMapping->lookup(targetKey);
                    if (taxonomyHook->expression[thread_idx]->isAncestor(currTax) == false) {
                        continue;
                    }
                }

                const bool isIdentity = (queryKey == targetKey && (par.includeIdentity || sameDB))? true : false;
                if(sequenceLookup == NULL){
                    char * targetSeq = tdbr->getData(tId, thread_idx);
                    unsigned int targetSeqLen = tdbr->getSeqLen(tId);
                    tSeq.mapSequence(tId, targetKey, targetSeq, targetSeqLen);
                    // Soft-mask remap: mmseqs stores soft-masked residues offset by +32 and
                    // lowercase-masked ones as ASCII 'a'..'z', so both ranges are folded to X here.
                    // A packed 3Di+12st DB has no such convention -- its bytes use the whole
                    // 0..251 range (state3di * 12 + state12st), so 32..52 and >= 97 are ordinary
                    // states. Applying this to a packed DB masks ~84% of all target residues to X
                    // and destroys the alignment. Same failure mode as maskLowerCaseLetter on the
                    // GPU path; skip it whenever the target carries the aux channel.
                    if (targetHasAux == false) {
                        unsigned char xChar = subMat->aa2num[static_cast<int>('X')];
                        for (int i = 0; i < tSeq.L; i++) {
                            tSeq.numSequence[i] = ((targetSeq[i] >= 32 && targetSeq[i] <= 52) || targetSeq[i] >= 97)  ? xChar : tSeq.numSequence[i];
                        }
                    }
                }else{
                    tSeq.mapSequence(tId, targetKey, sequenceLookup->getSequence(tId));
                }
                float queryLength = qSeq.L;
                float targetLength = tSeq.L;
                if(Util::canBeCovered(par.covThr, par.covMode, queryLength, targetLength)==false){
                    continue;
                }

                bool hasEvalue = true;
                int score;
                int best3DiDiagonal = 0;
                if (alignmentMode == 0) {
                    if (use12St) {
                        // ask for the diagonal too: the 12st rescore needs the alignment this pass
                        // found, and recovering it afterwards would mean rescanning the matrix
                        score = aligner.ungapped_alignment(tSeq.numSequence, tSeq.L, best3DiDiagonal);
                    } else {
                        score = aligner.ungapped_alignment(tSeq.numSequence, tSeq.L);
                    }
                } else {
                    std::string backtrace;
                    s_align res;
                    if (isIdentity) {
                        res = aligner.scoreIdentical(
                                tSeq.numSequence, tSeq.L, evaluer, Matcher::SCORE_ONLY, backtrace
                        );
                    } else {
                        res = aligner.ssw_align(
                                tSeq.numSequence,
                                tSeq.L,
                                backtrace,
                                par.gapOpen.values.aminoacid(),
                                par.gapExtend.values.aminoacid(),
                                Matcher::SCORE_ONLY,
                                par.evalThr,
                                evaluer,
                                par.covMode,
                                par.covThr,
                                par.correlationScoreWeight,
                                qSeq.L / 2
                        );
                    }
                    score = res.score1;
                    // check if evalThr != inf
                    double evalue = 0.0;
                    if (par.evalThr < std::numeric_limits<double>::max()) {
                        evalue = evaluer->computeEvalue(score, qSeq.L);
                    }
                    hasEvalue = (evalue <= par.evalThr);
                }
                // With the 12st channel on, --min-ungapped-score has to be applied to the *combined*
                // score, as it is on the GPU (where libmarv already folded 12st in before
                // par.minDiagScoreThr is tested). Applying it here to the 3Di-only score would
                // discard pairs the 12st term would have lifted over the threshold. Phase 1
                // therefore only uses a >0 floor and the real threshold is applied after the
                // rescore; the floor keeps the candidate set bounded to targets with some 3Di signal.
                bool hasDiagScore = use12St ? (score > 0) : (score > par.minDiagScoreThr);
                if (isIdentity || (hasDiagScore && hasEvalue)) {
                    hit_t hit;
                    hit.seqId = targetKey;
                    hit.prefScore = score;
                    // carry the 3Di diagonal through to the rescore (biased into unsigned short;
                    // it is only ever read back by the 12st rescore below)
                    hit.diagonal = use12St ? (unsigned short)(short)best3DiDiagonal : 0;
                    threadShortResults.emplace_back(hit);
                }
            }
#pragma omp critical
            {
                shortResults.insert(shortResults.end(), threadShortResults.begin(), threadShortResults.end());
                threadShortResults.clear();
            }
#pragma omp barrier
            if (use12St) {
                // rank by 3Di and decide how many hits get the 12st channel
#pragma omp master
                {
                    SORT_PARALLEL(shortResults.begin(), shortResults.end(), hit_t::compareHitsByScoreAndId);
                    rescoreCount = std::min(topN12St, shortResults.size());
                }
#pragma omp barrier
                // add the 12st channel to the top-K on the best 3Di diagonal
#pragma omp for schedule(dynamic, 16)
                for (size_t i = 0; i < rescoreCount; i++) {
                    const size_t tId = tdbr->getId(shortResults[i].seqId);
                    if (tId == UINT_MAX) {
                        continue;
                    }
                    if (sequenceLookup == NULL) {
                        tSeq.mapSequence(tId, shortResults[i].seqId,
                                         tdbr->getData(tId, thread_idx), tdbr->getSeqLen(tId));
                    } else {
                        tSeq.mapSequence(tId, shortResults[i].seqId, sequenceLookup->getSequence(tId));
                    }
                    if (tSeq.numSequenceAux == NULL || qSeq.numSequenceAux == NULL) {
                        continue;
                    }
                    const int diagonal = (int)(short)shortResults[i].diagonal;
                    shortResults[i].prefScore += score12StOnDiagonal(
                            qSeq.numSequenceAux, qSeq.L,
                            tSeq.numSequenceAux, tSeq.L,
                            subMat12St, diagonal);
                }
#pragma omp barrier
                // now apply --min-ungapped-score to the combined score (identity hits are always
                // kept, matching phase 1 and the GPU path)
#pragma omp master
                {
                    const unsigned int qKey = queryKey;
                    const bool keepIdentity = (par.includeIdentity || sameDB);
                    size_t kept = 0;
                    for (size_t i = 0; i < shortResults.size(); i++) {
                        const bool isIdentity = (qKey == shortResults[i].seqId) && keepIdentity;
                        if (isIdentity || shortResults[i].prefScore > par.minDiagScoreThr) {
                            shortResults[kept++] = shortResults[i];
                        }
                    }
                    shortResults.resize(kept);
                }
#pragma omp barrier
            }
#pragma omp master
            {
                SORT_PARALLEL(shortResults.begin(), shortResults.end(), hit_t::compareHitsByScoreAndId);
                size_t maxSeqs = std::min(par.maxResListLen, shortResults.size());
                for (size_t i = 0; i < maxSeqs; ++i) {
                    size_t len = QueryMatcher::prefilterHitToBuffer(buffer, shortResults[i]);
                    resultBuffer.append(buffer, len);
                }

                resultWriter.writeData(resultBuffer.c_str(), resultBuffer.length(), queryKey, 0);
                resultBuffer.clear();
                shortResults.clear();
                progress.updateProgress();
            }
#pragma omp barrier
        }
    }
}

int prefilterInternal(int argc, const char **argv, const Command &command, int mode) {
    Parameters &par = Parameters::getInstance();
    par.parseParameters(argc, argv, command, true, 0, 0);
    int outputDbtype = (par.prefMode == Parameters::PREF_MODE_UNGAPPED_AND_GAPPED)
                      ? Parameters::DBTYPE_ALIGNMENT_RES : Parameters::DBTYPE_PREFILTER_RES;
    DBWriter resultWriter(par.db3.c_str(), par.db3Index.c_str(), 1, par.compressed, outputDbtype);
    resultWriter.open();
    bool sameDB = (par.db2.compare(par.db1) == 0);
    bool touch = (par.preloadMode != Parameters::PRELOAD_MODE_MMAP);
    IndexReader tDbrIdx(par.db2, par.threads, IndexReader::SEQUENCES, (touch) ? (IndexReader::PRELOAD_INDEX | IndexReader::PRELOAD_DATA) : 0 );
    IndexReader * qDbrIdx = NULL;
    DBReader<unsigned int> * qdbr = NULL;
    DBReader<unsigned int> * tdbr = tDbrIdx.sequenceReader;

    if (par.gpu == true) {
        const bool isGpuDb = DBReader<unsigned int>::getExtendedDbtype(tdbr->getDbtype()) & Parameters::DBTYPE_EXTENDED_GPU;
        if (isGpuDb == false) {
            Debug(Debug::ERROR) << "Database " << FileUtil::baseName(par.db2) << " is not a valid GPU database\n" 
                                << "Please call: makepaddedseqdb " << FileUtil::baseName(par.db2) << " " << FileUtil::baseName(par.db2) << "_pad\n";
            EXIT(EXIT_FAILURE);
        }
    }

    const int targetSeqType = tdbr->getDbtype();
    int querySeqType;
    if (sameDB == true) {
        qDbrIdx = &tDbrIdx;
        qdbr = tdbr;
        querySeqType = targetSeqType;
    } else {
        // open the sequence, prefiltering and output databases
        qDbrIdx = new IndexReader(par.db1, par.threads, IndexReader::SEQUENCES, (touch) ? IndexReader::PRELOAD_INDEX : 0);
        qdbr = qDbrIdx->sequenceReader;
        querySeqType = qdbr->getDbtype();
    }

    SequenceLookup * sequenceLookup = NULL;
    if(Parameters::isEqualDbtype(tDbrIdx.getDbtype(), Parameters::DBTYPE_INDEX_DB)){
        PrefilteringIndexData data = PrefilteringIndexReader::getMetadata(tDbrIdx.index);
        if(data.splits == 1){
            sequenceLookup = PrefilteringIndexReader::getSequenceLookup(0, tDbrIdx.index, par.preloadMode);
        }
    }
    BaseMatrix *subMat;
    EvalueComputation * evaluer;
    int8_t * tinySubMat;
    if (Parameters::isEqualDbtype(querySeqType, Parameters::DBTYPE_NUCLEOTIDES)) {
        subMat = new NucleotideMatrix(par.scoringMatrixFile.values.nucleotide().c_str(), 1.0, 0.0);
        evaluer = new EvalueComputation(tdbr->getAminoAcidDBSize(), subMat, par.gapOpen.values.nucleotide(), par.gapExtend.values.nucleotide());
        tinySubMat = new int8_t[subMat->alphabetSize*subMat->alphabetSize];
        for (int i = 0; i < subMat->alphabetSize; i++) {
            for (int j = 0; j < subMat->alphabetSize; j++) {
                tinySubMat[i*subMat->alphabetSize + j] = subMat->subMatrix[i][j];
            }
        }
    } else {
        // keep score bias at 0.0 (improved ROC)
        subMat = new SubstitutionMatrix(par.scoringMatrixFile.values.aminoacid().c_str(), 2.0, 0.0);
        evaluer = new EvalueComputation(tdbr->getAminoAcidDBSize(), subMat, par.gapOpen.values.aminoacid(), par.gapExtend.values.aminoacid());
        tinySubMat = new int8_t[subMat->alphabetSize*subMat->alphabetSize];
        for (int i = 0; i < subMat->alphabetSize; i++) {
            for (int j = 0; j < subMat->alphabetSize; j++) {
                tinySubMat[i*subMat->alphabetSize + j] = subMat->subMatrix[i][j];
            }
        }
    }


    // Foldseek GPU 12-state: build the auxiliary 12st substitution matrix so the GPU prefilter
    // can add the 12st channel. Two conditions:
    //   * the target GPU DB carries the packed 3Di+12st flag (EXTENDED_3DI_12ST = 32, set by
    //     foldseek's LocalParameters and preserved through the padded-DB build), and
    //   * --aux-score is on. That is the same switch the k-mer prefilter uses, and the workflows
    //     drive it from --ss-12st (StructureSearch.cpp: par.useAuxScoring = par.ss12st), so
    //     --ss-12st 0 now disables the 12st channel on the GPU too instead of being ignored.
    // Either condition failing leaves subMat12St NULL. Note that for a *packed* DB that still
    // means the foldseek entry point is used (it is the only one that can read a packed byte) --
    // runFilterOnGpu just feeds it an all-zero 12st profile. A legacy 3Di-only GPU DB takes the
    // original single-channel path unchanged.
    // The matrix is built for the CPU path too: runFilterOnCpu runs the same
    // score-3Di-then-rescore-top-K workflow (see score12StOnDiagonal).
    const uint16_t EXT_3DI_12ST = 32;
    const bool packedTargetDb =
        (DBReader<unsigned int>::getExtendedDbtype(tdbr->getDbtype()) & EXT_3DI_12ST) != 0;
    const bool packedTarget = packedTargetDb && par.useAuxScoring;
    if (packedTargetDb && par.useAuxScoring == false) {
        Debug(Debug::INFO) << "12-state channel disabled by --aux-score/--ss-12st; "
                           << "GPU prefilter scores 3Di only\n";
    }
    BaseMatrix *subMat12St = NULL;
    if (packedTarget) {
        std::string mat12st;
        for (size_t i = 0; i < par.substitutionMatrices.size(); i++) {
            if (par.substitutionMatrices[i].name == "12st.out") {
                std::string matrixData((const char *)par.substitutionMatrices[i].subMatData,
                                       par.substitutionMatrices[i].subMatDataLen);
                char *serializedMatrix = BaseMatrix::serialize(par.substitutionMatrices[i].name, matrixData);
                mat12st.assign(serializedMatrix);
                free(serializedMatrix);
                break;
            }
        }
        if (mat12st.empty()) {
            Debug(Debug::ERROR) << "Cannot find 12st substitution matrix for the GPU 12-state prefilter\n";
            EXIT(EXIT_FAILURE);
        }
        // Bit factor 2.0 and scoreBias -0.2 mirror the CPU k-mer prefilter's aux 12st matrix
        // (Prefiltering.cpp: `new SubstitutionMatrix(serialized, 2.0, -0.2f)`), so the GPU and CPU
        // prefilters score the 12st channel identically. 3Di uses the same 2.0 bit factor here,
        // keeping the two channels balanced 1:1 as in structurealign (2.1:2.1).
        subMat12St = new SubstitutionMatrix(mat12st.c_str(), 2.0, -0.2f);
    }

    QueryMatcherTaxonomyHook * taxonomyHook = NULL;
    if(par.PARAM_TAXON_LIST.wasSet){
        taxonomyHook = new QueryMatcherTaxonomyHook(par.db2, tdbr, par.taxonList, par.threads);
    }
    if(par.gpu){
#ifdef HAVE_CUDA
        runFilterOnGpu(par, subMat, subMat12St, qdbr, tdbr, sameDB,
                       resultWriter, evaluer, taxonomyHook);
#else
        Debug(Debug::ERROR) << "MMseqs2 was compiled without CUDA support\n";
        EXIT(EXIT_FAILURE);
#endif
    }else{
        runFilterOnCpu(par, subMat, subMat12St, tinySubMat, qdbr, tdbr, sequenceLookup, sameDB,
                   resultWriter, evaluer, taxonomyHook,  mode);
    }

    resultWriter.close();

    if(taxonomyHook != NULL){
        delete taxonomyHook;
    }

    if (sequenceLookup != NULL) {
        delete sequenceLookup;
    }

    if(sameDB == false){
        delete qDbrIdx;
    }

    delete [] tinySubMat;
    delete subMat;
    if (subMat12St != NULL) {
        delete subMat12St;
    }
    delete evaluer;

    return 0;
}

int ungappedprefilter(int argc, const char **argv, const Command &command) {
    return prefilterInternal(argc, argv, command, 0);
}

int gappedprefilter(int argc, const char **argv, const Command &command) {
    return prefilterInternal(argc, argv, command, 1);
}
