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
static const char EVALUE_12ST_THREEDI_ALPHABET[] = {
        'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
        'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y', 'X'
};
static const char EVALUE_12ST_ALPHABET[] = {
        'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'X'
};
static const int EVALUE_12ST_THREEDI_ALPHABET_SIZE = sizeof(EVALUE_12ST_THREEDI_ALPHABET) / sizeof(EVALUE_12ST_THREEDI_ALPHABET[0]);
static const int EVALUE_12ST_ALPHABET_SIZE = sizeof(EVALUE_12ST_ALPHABET) / sizeof(EVALUE_12ST_ALPHABET[0]);
}


EvalueNeuralNet::EvalueNeuralNet(size_t dbResidueCount,
                                 size_t dbSequenceCount,
                                 BaseMatrix* subMat,
                                 int mode,
                                 bool recoverProfileComposition,
                                 const BaseMatrix *subMat12St)
                                 : subMat(subMat), mode(mode), recoverProfileComposition(recoverProfileComposition), subMat12St(subMat12St) {
        logDbResidueCount = log(static_cast<double>(dbResidueCount));
        logDbSequenceCount = log(static_cast<double>(dbSequenceCount));
        const double uRef = log(EVALUE_12ST_LENGTH_REF);
        lengthFactorRef = (EVALUE_12ST_LENGTH_C2 * uRef + EVALUE_12ST_LENGTH_C1) * uRef;
        if (uses12StModel()) {
            encoder.LoadModel(
                    std::string((const char *)evalue_12st_nn_kerasify,
                                evalue_12st_nn_kerasify_len));
            in = Tensor(EVALUE_12ST_FEATURE_SIZE);
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

void EvalueNeuralNet::fill12StComposition(float *dst, const Sequence &seq, const char *alphabet, int alphabetSize, const BaseMatrix *backgroundMat) const {
    std::fill(dst, dst + alphabetSize, 0.0f);
    if (seq.L <= 0) {
        return;
    }
    if (recoverProfileComposition
        && Parameters::isEqualDbtype(seq.getSequenceType(), Parameters::DBTYPE_HMM_PROFILE)) {
        const short *profileScores = seq.profile_score;
        const size_t rowSize = seq.profile_row_size;
        float col[EVALUE_12ST_FEATURE_SIZE];
        for (int pos = 0; pos < seq.L; ++pos) {
            const short *row = profileScores + (static_cast<size_t>(pos) * rowSize);
            float colSum = 0.0f;
            for (int aa = 0; aa < alphabetSize; ++aa) {
                const unsigned char letter = static_cast<unsigned char>(alphabet[aa]);
                const unsigned char state = seq.subMat->aa2num[letter];
                float prob = 0.0f;
                if (state < static_cast<unsigned char>(Sequence::PROFILE_AA_SIZE)) {
                    prob = Sequence::scoreToProba(row[state], backgroundMat->pBack[backgroundMat->aa2num[letter]],
                                                  EVALUE_12ST_PROFILE_BIT_FACTOR, seq.subMat->scoreBias);
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

    // feature index of every state of seq's alphabet, -1 for states without one
    int feature[256];
    std::fill(feature, feature + 256, -1);
    for (int aa = 0; aa < alphabetSize; ++aa) {
        feature[seq.subMat->aa2num[static_cast<unsigned char>(alphabet[aa])]] = aa;
    }
    const float invLen = 1.0f / static_cast<float>(seq.L);
    for (int pos = 0; pos < seq.L; ++pos) {
        const int f = feature[seq.numSequence[pos]];
        if (f >= 0) {
            dst[f] += invLen;
        }
    }
}

std::pair<double, double> EvalueNeuralNet::predictMuLambda12St(const Sequence &seq3Di, const Sequence &seq12St) {
    std::fill(in.data_.begin(), in.data_.end(), 0.0f);
    fill12StComposition(in.data_.data(), seq3Di, EVALUE_12ST_THREEDI_ALPHABET, EVALUE_12ST_THREEDI_ALPHABET_SIZE, seq3Di.subMat);
    // a 12-state profile is laid out on the amino-acid matrix, but its log-odds are relative to the 12-state background
    fill12StComposition(in.data_.data() + EVALUE_12ST_THREEDI_ALPHABET_SIZE,
                        seq12St, EVALUE_12ST_ALPHABET, EVALUE_12ST_ALPHABET_SIZE,
                        subMat12St != NULL ? subMat12St : seq12St.subMat);
    in.data_[EVALUE_12ST_FEATURE_SIZE - 1] = static_cast<float>(seq3Di.L);
    encoder.Apply(&in, &out);
    return std::make_pair(std::max(out.data_[0] * EVALUE_12ST_LAMBDA_STD + EVALUE_12ST_LAMBDA_MEAN, static_cast<double>(EVALUE_12ST_MIN_LAMBDA)),
                          out.data_[1] * EVALUE_12ST_MU_STD + EVALUE_12ST_MU_MEAN);
}
