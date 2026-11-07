//
// Created by Martin Steinegger on 19/01/2022.
//

#include "EvalueNeuralNet.h"
#include "Sequence.h"
#include "evalue_nn.kerasify.h"
#include "evalue_12st_nn.kerasify.h"
#include <algorithm>
#include <numeric>

namespace {
static const char LEGACY_12ST_THREEDI_ALPHABET[] = {
        'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
        'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y', 'X'
};
static const char LEGACY_12ST_ALPHABET[] = {
        'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'X'
};
static const int LEGACY_12ST_THREEDI_ALPHABET_SIZE = sizeof(LEGACY_12ST_THREEDI_ALPHABET) / sizeof(LEGACY_12ST_THREEDI_ALPHABET[0]);
static const int LEGACY_12ST_ALPHABET_SIZE = sizeof(LEGACY_12ST_ALPHABET) / sizeof(LEGACY_12ST_ALPHABET[0]);
}


EvalueNeuralNet::EvalueNeuralNet(size_t dbResidueCount,
                                 size_t dbSequenceCount,
                                 BaseMatrix* subMat,
                                 int mode,
                                 bool,
                                 bool recoverProfileComposition,
                                 const BaseMatrix *subMat12St)
                                 : subMat(subMat), mode(mode), recoverProfileComposition(recoverProfileComposition), subMat12St(subMat12St) {
        logDbResidueCount = log(static_cast<double>(dbResidueCount));
        logDbSequenceCount = log(static_cast<double>(dbSequenceCount));
        if (usesLegacy12StModel()) {
            encoder.LoadModel(
                    std::string((const char *)evalue_12st_nn_kerasify,
                                evalue_12st_nn_kerasify_len));
            in = Tensor(LEGACY_12ST_FEATURE_SIZE);
            out = Tensor(2);
        } else {
            encoder.LoadModel(
            std::string((const char *)evalue_nn_kerasify,
            evalue_nn_kerasify_len));
            in = Tensor(subMat->alphabetSize + 1);
            out = Tensor(2);
        }
}

std::pair<double, double> EvalueNeuralNet::predictMuLambda(const unsigned char * seq, unsigned int L){
    for(int i = 0; i < subMat->alphabetSize; i++){
        in.data_[i] = 0;
    }
    for (unsigned int i = 0; i < L; i++) {
        in.data_[seq[i]]++;
    }
    in.data_[subMat->alphabetSize] = L;
    encoder.Apply(&in, &out);
    // used to normalize the output
    double mu1 = 0.17518475184751847;
    double sigma1 = 0.03260331312698818;
    double mu2 = -2.5569312493124934;
    double sigmal2 = 0.4353169278257701;
    return std::make_pair(out.data_[0]*sigma1+mu1,
                          out.data_[1]*sigmal2+mu2);
}

std::pair<double, double> EvalueNeuralNet::predictMuLambda(const Sequence &seq){
    return predictMuLambda(seq.numSequence, seq.L);
}

std::pair<double, double> EvalueNeuralNet::predictMuLambda(const Sequence &seq3Di, const Sequence &seq12St) {
    if (!usesLegacy12StModel()) {
        return predictMuLambda(seq3Di);
    }
    return predictMuLambdaLegacy12St(seq3Di, seq12St);
}

void EvalueNeuralNet::fillLegacy12StComposition(float *dst, const Sequence &seq, const char *alphabet, int alphabetSize, const BaseMatrix *backgroundMat) const {
    std::fill(dst, dst + alphabetSize, 0.0f);
    if (seq.L <= 0) {
        return;
    }
    if (recoverProfileComposition
        && Parameters::isEqualDbtype(seq.getSequenceType(), Parameters::DBTYPE_HMM_PROFILE)) {
        const short *profileScores = seq.profile_score;
        const size_t rowSize = seq.profile_row_size;
        std::vector<float> col(alphabetSize);
        for (int pos = 0; pos < seq.L; ++pos) {
            const short *row = profileScores + (static_cast<size_t>(pos) * rowSize);
            float colSum = 0.0f;
            for (int aa = 0; aa < alphabetSize; ++aa) {
                const unsigned char letter = static_cast<unsigned char>(alphabet[aa]);
                const unsigned char state = seq.subMat->aa2num[letter];
                float prob = 0.0f;
                if (state < static_cast<unsigned char>(Sequence::PROFILE_AA_SIZE)) {
                    prob = Sequence::scoreToProba(row[state], backgroundMat->pBack[backgroundMat->aa2num[letter]],
                                                  LEGACY_12ST_PROFILE_BIT_FACTOR, seq.subMat->scoreBias);
                }
                col[aa] = prob;
                colSum += prob;
            }
            if (colSum > 0.0f) {
                const float invColSum = 1.0f / colSum;
                for (int aa = 0; aa < alphabetSize; ++aa) {
                    dst[aa] += col[aa] * invColSum;
                }
            }
        }
        const float invLen = 1.0f / static_cast<float>(seq.L);
        for (int aa = 0; aa < alphabetSize; ++aa) {
            dst[aa] *= invLen;
        }
        return;
    }

    const float invLen = 1.0f / static_cast<float>(seq.L);
    for (int aa = 0; aa < alphabetSize; ++aa) {
        const unsigned char state = seq.subMat->aa2num[static_cast<unsigned char>(alphabet[aa])];
        for (int pos = 0; pos < seq.L; ++pos) {
            if (seq.numSequence[pos] == state) {
                dst[aa] += invLen;
            }
        }
    }
}

std::pair<double, double> EvalueNeuralNet::predictMuLambdaLegacy12St(const Sequence &seq3Di, const Sequence &seq12St) {
    std::fill(in.data_.begin(), in.data_.end(), 0.0f);
    fillLegacy12StComposition(in.data_.data(), seq3Di, LEGACY_12ST_THREEDI_ALPHABET, LEGACY_12ST_THREEDI_ALPHABET_SIZE, seq3Di.subMat);
    // a 12-state profile is laid out on the amino-acid matrix, but its log-odds are relative to the 12-state background
    fillLegacy12StComposition(in.data_.data() + LEGACY_12ST_THREEDI_ALPHABET_SIZE,
                              seq12St, LEGACY_12ST_ALPHABET, LEGACY_12ST_ALPHABET_SIZE,
                              subMat12St != NULL ? subMat12St : seq12St.subMat);
    in.data_[LEGACY_12ST_FEATURE_SIZE - 1] = static_cast<float>(seq3Di.L);
    encoder.Apply(&in, &out);
    return std::make_pair(std::max(out.data_[0] * LEGACY_12ST_LAMBDA_STD + LEGACY_12ST_LAMBDA_MEAN, static_cast<double>(LEGACY_12ST_MIN_LAMBDA)),
                          out.data_[1] * LEGACY_12ST_MU_STD + LEGACY_12ST_MU_MEAN);
}
