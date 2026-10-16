#ifndef STRUCTURE_UTIL_H
#define STRUCTURE_UTIL_H

#include "Util.h"
#include "PrefilteringIndexReader.h"
#include "LocalParameters.h"
#include "BaseMatrix.h"
#include "structureto12st.h"
#include "Sequence.h"

#include <vector>
#include <atomic>
#include "Debug.h"
#include <cstdlib>
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
    static inline void mask12StSeqs(char *seq3Di, char *seq12St, size_t len, bool isQuery) {
        if (!isQuery) { // only mask the query sequence
            return;
        }
        for (size_t i = 0; i < len; i++) {
            if ((seq3Di[i] == 'D' || seq3Di[i] == 'V' || seq3Di[i] == 'P') &&
                (seq12St[i] == 'G' || seq12St[i] == 'H' || seq12St[i] == 'I')) {
                seq12St[i] = 'X';
            }
        }
    }

    static inline bool profileMaskingEnabled() {
        static const bool on = [] {
            const char *v = std::getenv("FOLDSEEK_NO_PROFILE_MASK");
            return !(v != NULL && v[0] != '\0' && v[0] != '0');
        }();
        return on;
    }
    static std::atomic<size_t> &profileMaskedPositions() { static std::atomic<size_t> v(0); return v; }
    static std::atomic<size_t> &profileMaskTotalPositions() { static std::atomic<size_t> v(0); return v; }
    static void reportProfileMasking(const char *tool) {
        const size_t tot = profileMaskTotalPositions().load();
        if (tot == 0) return;
        const size_t m = profileMaskedPositions().load();
        Debug(Debug::INFO) << tool << ": profile queries, 12st D/V/P x G/H/I mask on " << m << " of " << tot
                           << " positions (" << (100.0 * m / tot) << "%)\n";
    }
    static inline size_t mask12StProfile(const Sequence &q3Di, Sequence &q12St) {
        if (!profileMaskingEnabled()) {
            return 0;
        }
        const int L = std::min(q3Di.L, q12St.L);
        const int alph = q12St.subMat->alphabetSize;
        size_t masked = 0;
        for (int i = 0; i < L; i++) {
            const char c3 = q3Di.subMat->num2aa[q3Di.numSequence[i]];
            const char c12 = q12St.subMat->num2aa[q12St.numSequence[i]];
            if ((c3 == 'D' || c3 == 'V' || c3 == 'P') && (c12 == 'G' || c12 == 'H' || c12 == 'I')) {
                for (int a = 0; a < alph; a++) {
                    q12St.profile_for_alignment[(size_t) a * q12St.L + i] = 0;
                }
                masked++;
            }
        }
        profileMaskedPositions() += masked;
        profileMaskTotalPositions() += (size_t) L;
        return masked;
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
