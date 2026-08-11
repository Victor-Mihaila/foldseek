#include "Parameters.h"
#include "DBReader.h"
#include "DBWriter.h"
#include "Debug.h"
#include "Util.h"
#include "SubstitutionMatrix.h"
#include "tantan.h"
#include "Masker.h"

#ifdef OPENMP
#include <omp.h>
#endif

int makepaddedseqdb(int argc, const char **argv, const Command &command) {
    Parameters &par = Parameters::getInstance();
    par.parseParameters(argc, argv, command, true, 0, 0);

    const int mode = DBReader<unsigned int>::USE_INDEX | DBReader<unsigned int>::USE_DATA;
    DBReader<unsigned int> dbr(par.db1.c_str(), par.db1Index.c_str(), par.threads, mode);
    dbr.open(DBReader<unsigned int>::SORT_BY_LENGTH);

    DBReader<unsigned int> dbhr(par.hdr1.c_str(), par.hdr1Index.c_str(), par.threads, mode);
    dbhr.open(DBReader<unsigned int>::NOSORT);

    SubstitutionMatrix subMat(par.scoringMatrixFile.values.aminoacid().c_str(), 2.0, par.scoreBias);

    int dbType = DBReader<unsigned int>::setExtendedDbtype(dbr.getDbtype(), Parameters::DBTYPE_EXTENDED_GPU);

    // Foldseek GPU 12-state layout: when the input _ss DB packs both structural alphabets into
    // a single byte (state3di * Alphabet12St::STATE_CNT + state12st, 12st in 0..11), keep that
    // byte intact and unmasked so the GPU kernel can split it (byte/12, byte%12) and apply
    // --mask-n-repeat masking on the fly. Detected via the extended dbtype bit that foldseek's
    // LocalParameters::DBTYPE_EXTENDED_3DI_12ST sets (= 32); that bit is preserved into dbType
    // above by setExtendedDbtype's OR, so the GPU driver can detect the packed layout on the
    // output DB as well. Non-packed (legacy 3Di-only) DBs fall through to the original path.
    const uint16_t EXT_3DI_12ST = 32;
    const bool packed3Di12St =
        (DBReader<unsigned int>::getExtendedDbtype(dbr.getDbtype()) & EXT_3DI_12ST) != 0;
    // Packed byte used for the SIMD tail padding after each sequence.
    //
    // This must NOT be structcreatedb's invalid encoding (Alphabet3Di::INVALID_STATE = 2,
    // Alphabet12St::INVALID_STATE = 6 -> byte 30). Those are deliberate biological fallbacks for
    // *unresolved residues* -- 3Di state 2 is the coil state, which scores like any other state.
    // Padding is not a residue at all, and the legacy 3Di-only path pads with 'X' (numeric 20)
    // exactly so the tail scores neutrally. Using state 2 here perturbs the gapless scores of the
    // padded tail relative to the legacy path.
    //
    // So pad with 3Di 'X' in the high channel: 20 * STATE_CNT + Alphabet12St::INVALID_STATE.
    // The low (12st) channel of the padding is never read -- libmarv's 12st diagonal rescore is
    // bounded by the real sequence length -- so its value only needs to stay in 0..11.
    const unsigned char PACKED_PAD_3DI_X = 20 * 12 + 6; // = 246 -> 3Di 20 ('X'), 12st 6

    DBWriter dbsw(par.db2.c_str(), par.db2Index.c_str(), par.threads, false, dbType);
    dbsw.open();
    DBWriter dbhw(par.hdr2.c_str(), par.hdr2Index.c_str(), par.threads, false, Parameters::DBTYPE_GENERIC_DB);
    dbhw.open();

    // need to prune low scoring k-mers through masking

    Debug::Progress progress(dbr.getSize());
#pragma omp parallel
{
    unsigned int thread_idx = 0;
#ifdef OPENMP
    thread_idx = static_cast<unsigned int>(omp_get_thread_num());
#endif
    Masker masker(subMat);
    std::string result;
    result.reserve(par.maxSeqLen);

    const int ALIGN = 4;
    Sequence seq(dbr.getMaxSeqLen(), dbr.getDbtype(), &subMat,  0, false, false);

    size_t firstIt = SIZE_MAX;
    unsigned int seqKey = 0;

    size_t charSeqBufferSize = par.maxSeqLen + 1;
    unsigned char *charSequence = NULL;
    if (par.maskMode) {
        charSequence = (unsigned char*)malloc(charSeqBufferSize * sizeof(char));
    }

#pragma omp for schedule(static)
    for (size_t i = 0; i < dbr.getSize(); i++) {
        progress.updateProgress();

        if (firstIt == SIZE_MAX) {
            firstIt = i;
        }

        size_t id = dbr.getSize() - 1 - i;
        unsigned int key = dbr.getDbKey(id);
        char *data = dbr.getData(id, thread_idx);
        size_t seqLen = dbr.getSeqLen(id);

        size_t writtenLen;
        char padByte;
        if (packed3Di12St) {
            // Ship the raw packed (state3di*12 + state12st) byte unmasked; the GPU kernel
            // splits and masks on the fly. seqLen comes from the index (getSeqLen), never
            // newline scanning, which is required because a packed byte can equal '\n' (0x0A).
            result.append(data, seqLen);
            writtenLen = seqLen;
            padByte = static_cast<char>(PACKED_PAD_3DI_X);
        } else {
            seq.mapSequence(id, key, data, seqLen);
            if (charSequence != NULL) {
                if ((size_t)seq.L >= charSeqBufferSize) {
                    charSeqBufferSize = seq.L * 1.5;
                    charSequence = (unsigned char*)realloc(charSequence, charSeqBufferSize * sizeof(char));
                }
                memcpy(charSequence, seq.numSequence, seq.L);
                masker.maskSequence(seq, par.maskMode, par.maskProb, par.maskLowerCaseMode, par.maskNrepeats);
                for (int j = 0; j < seq.L; j++) {
                    result.append(1, (seq.numSequence[j] == masker.maskLetterNum) ? charSequence[j] + 32 : charSequence[j]);
                }
            } else {
                for (int j = 0; j < seq.L; j++) {
                    char aa = data[j];
                    result.append(1, (islower(aa)) ? seq.numSequence[j] + 32 : seq.numSequence[j]);
                }
            }
            writtenLen = seq.L;
            padByte = static_cast<char>(20);
        }
        const size_t sequencepadding = (writtenLen % ALIGN == 0) ? 0 : ALIGN - writtenLen % ALIGN;
        result.append(sequencepadding, padByte);
        dbsw.writeData(result.c_str(), result.size(), key, thread_idx, false, false);

        // + 2 is needed for newline and null character
        size_t start = dbsw.getStart(thread_idx);
        if (start % 4 != 0) {
            Debug(Debug::ERROR) << "Misalligned entry\n";
            EXIT(EXIT_FAILURE);
        }
        dbsw.writeIndexEntry(firstIt + seqKey, start, writtenLen + 2, thread_idx);

        unsigned int headerId = dbhr.getId(key);
        dbhw.writeData(dbhr.getData(headerId, thread_idx), dbhr.getEntryLen(headerId), firstIt + seqKey, thread_idx, false);

        seqKey++;
        result.clear();
    }
    if (charSequence != NULL) {
        free(charSequence);
    }
}
    dbsw.close(true, false);
    dbhw.close(true, false);
    dbhr.close();
    if (par.writeLookup == true) {
        DBReader<unsigned int> readerHeader(par.hdr2.c_str(), par.hdr2Index.c_str(), 1, DBReader<unsigned int>::USE_DATA | DBReader<unsigned int>::USE_INDEX);
        readerHeader.open(DBReader<unsigned int>::NOSORT);
        // create lookup file
        std::string lookupFile = par.db2 + ".lookup";
        FILE* file = FileUtil::openAndDelete(lookupFile.c_str(), "w");
        std::string buffer;
        buffer.reserve(2048);
        DBReader<unsigned int>::LookupEntry entry;
        size_t totalSize = dbr.getSize();
        for (unsigned int id = 0; id < readerHeader.getSize(); id++) {
            char *header = readerHeader.getData(id, 0);
            entry.id = id;
            entry.entryName = Util::parseFastaHeader(header);
            entry.fileNumber = dbr.getDbKey(totalSize - 1 - id);
            readerHeader.lookupEntryToBuffer(buffer, entry);
            int written = fwrite(buffer.c_str(), sizeof(char), buffer.size(), file);
            if (written != (int)buffer.size()) {
                Debug(Debug::ERROR) << "Cannot write to lookup file " << lookupFile << "\n";
                EXIT(EXIT_FAILURE);
            }
            buffer.clear();
        }
        if (fclose(file) != 0) {
            Debug(Debug::ERROR) << "Cannot close file " << lookupFile << "\n";
            EXIT(EXIT_FAILURE);
        }
        readerHeader.close();
    }
    dbr.close();
    return EXIT_SUCCESS;
}