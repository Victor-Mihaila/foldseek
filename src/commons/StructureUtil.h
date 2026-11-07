#ifndef STRUCTURE_UTIL_H
#define STRUCTURE_UTIL_H

#include "Util.h"
#include "PrefilteringIndexReader.h"
#include "LocalParameters.h"
#include "BaseMatrix.h"
#include "structureto12st.h"
#include "Sequence.h"

#include <vector>
#include "Debug.h"
#include <algorithm>

class StructureUtil {
public:
    static bool is3Di12StDb(int dbtype) {
        return (DBReader<unsigned int>::getExtendedDbtype(dbtype)
                & Parameters::DBTYPE_EXTENDED_AUX_SEQ) != 0;
    }

    // Mask redundant 12st positions for query sequences only.
    // 3Di states D/V/P and 12st states G/H/I both indicate "no structural
    // neighbor found", so they encode the same lack-of-contact signal.
    // Masking the 12st character to 'X' at these positions removes the
    // redundancy.
    static inline bool isRedundant12StPosition(char c3Di, char c12St) {
        return (c3Di == 'D' || c3Di == 'V' || c3Di == 'P') && (c12St == 'G' || c12St == 'H' || c12St == 'I');
    }

    static inline void mask12StSeqs(char *seq3Di, char *seq12St, size_t len, bool isQuery) {
        if (!isQuery) { // only mask the query sequence
            return;
        }
        for (size_t i = 0; i < len; i++) {
            if (isRedundant12StPosition(seq3Di[i], seq12St[i])) {
                seq12St[i] = 'X';
            }
        }
    }

    // profile counterpart of mask12StSeqs: zero the 12st profile column at the same positions
    static inline void mask12StProfile(const Sequence &q3Di, Sequence &q12St) {
        const int L = std::min(q3Di.L, q12St.L);
        const int alph = q12St.subMat->alphabetSize;
        for (int i = 0; i < L; i++) {
            if (isRedundant12StPosition(q3Di.subMat->num2aa[q3Di.numSequence[i]], q12St.subMat->num2aa[q12St.numSequence[i]])) {
                for (int a = 0; a < alph; a++) {
                    q12St.profile_for_alignment[(size_t) a * q12St.L + i] = 0;
                }
            }
        }
    }

    static inline void split3Di12St(const char *src, size_t len,
                                     std::vector<char> &seq3di,
                                     std::vector<char> &seq12st,
                                     const BaseMatrix &subMat3Di,
                                     const BaseMatrix &subMat12St,
                                     const bool isQuery=false) {
        if (seq3di.size() < len) {
            seq3di.resize(len);
        }
        if (seq12st.size() < len) {
            seq12st.resize(len);
        }
        for (size_t i = 0; i < len; ++i) {
            unsigned char val = static_cast<unsigned char>(src[i]);
            if (val == Alphabet12St::PACKED_X) {
                seq3di[i] = 'X';
                seq12st[i] = 'X';
                continue;
            }
            unsigned char state3di = static_cast<unsigned char>(val / Alphabet12St::STATE_CNT);
            unsigned char state12st = static_cast<unsigned char>(val % Alphabet12St::STATE_CNT);
            seq3di[i] = subMat3Di.num2aa[state3di];
            seq12st[i] = subMat12St.num2aa[state12st];
        }
        mask12StSeqs(seq3di.data(), seq12st.data(), len, isQuery);
    }

    static std::string getIndexWithSuffix(std::string db, const std::string &suffix) {
        if (Util::endsWith(".idx", db)) {
            db = db.substr(0, db.length() - 4);
        } else {
            return db + suffix;
        }
        db.append(suffix);
        std::string index = PrefilteringIndexReader::searchForIndex(db);
        if (index != "") {
            return index;
        }
        return db;
    }
};

#endif
