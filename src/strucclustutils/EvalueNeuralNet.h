//
// Created by Martin Steinegger on 19/01/2022.
//

#ifndef FOLDSEEK_EVALUENEURALNET_H
#define FOLDSEEK_EVALUENEURALNET_H
#include <algorithm>
#include <cmath>
#include <vector>
#include "kerasify/keras_model.h"
#include "BaseMatrix.h"
#include "LocalParameters.h"

class Sequence;

class EvalueNeuralNet {
private:
    static constexpr double EVALUE_12ST_CORR_SCALE = 0.1357792528025477;
    static constexpr double EVALUE_12ST_LENGTH_REF = 150.0;
    static constexpr double EVALUE_12ST_LENGTH_C1 = -1.8318504160819078;
    static constexpr double EVALUE_12ST_LENGTH_C2 = 0.45780993271377185;
    static constexpr double EVALUE_12ST_LENGTH_MIN = 60.0;
    static constexpr double EVALUE_12ST_LENGTH_MAX = 400.0;
    static constexpr int EVALUE_12ST_FEATURE_SIZE = 35;
    static constexpr float EVALUE_12ST_PROFILE_BIT_FACTOR = 8.0f;
    static constexpr double EVALUE_12ST_LAMBDA_MEAN = 0.05809067;
    static constexpr double EVALUE_12ST_LAMBDA_STD = 0.03113629892;
    static constexpr double EVALUE_12ST_MU_MEAN = 7.340611654;
    static constexpr double EVALUE_12ST_MU_STD = 7.853373165;
    static constexpr double EVALUE_12ST_MIN_LAMBDA = 1e-4;
    BaseMatrix *subMat;
    double logDbResidueCount;
    double logDbSequenceCount;
    double lengthFactorRef;
    int mode;
    bool recoverProfileComposition;
    const BaseMatrix *subMat12St;
    KerasModel encoder;
    Tensor in;
    Tensor out;

    void fill12StComposition(float *dst, const Sequence &seq, const char *alphabet, int alphabetSize, const BaseMatrix *backgroundMat) const;
public:

    EvalueNeuralNet(size_t dbResidueCount,
                    size_t dbSequenceCount,
                    BaseMatrix* subMat,
                    int mode = LocalParameters::EVALUE_NN_MODE_3DI,
                    bool recoverProfileComposition = false,
                    const BaseMatrix *subMat12St = NULL);

    std::pair<double, double> predictMuLambda(const unsigned char * seq, unsigned int L);
    std::pair<double, double> predictMuLambda12St(const Sequence &seq3Di, const Sequence &seq12St);
    bool uses12StModel() const { return mode == LocalParameters::EVALUE_NN_MODE_3DI_12ST; }
    double activeLogDbCount() const { return uses12StModel() ? logDbSequenceCount : logDbResidueCount; }

    double computePvalue(double score, double lambda_, double mu) {
        double h = lambda_ * (score - mu);
        if(h > 10) {
            return -h;
        } else if (h < -2.5) {
            return -exp(-exp(-h));
        } else {
            return log((1.0 - exp(-exp(-h))));
        }
    }

    double computeEvalue(double score, double lambda_, double mu){
        return exp(computePvalue(score, lambda_, mu) + activeLogDbCount());
    }

    double computeEvalueCorr(double score, double lambda_, double mu,
                             unsigned int queryLen, unsigned int targetLen,
                             const std::pair<double, double> *targetMuLambda = NULL){
        double logPVal = computePvalue(score, lambda_, mu);
        if (uses12StModel() && targetMuLambda != NULL) {
            logPVal = 0.5 * (logPVal + computePvalue(score, targetMuLambda->first, targetMuLambda->second));
        }
        double dbSizeTimesLogPVal = logPVal + activeLogDbCount();
        double evalue = exp(dbSizeTimesLogPVal);
        if (uses12StModel()) {
            double u = log(std::min(std::max(static_cast<double>(std::min(queryLen, targetLen)), static_cast<double>(EVALUE_12ST_LENGTH_MIN)), static_cast<double>(EVALUE_12ST_LENGTH_MAX)));
            double logLengthFactor = (EVALUE_12ST_LENGTH_C2 * u + EVALUE_12ST_LENGTH_C1) * u - lengthFactorRef;
            return EVALUE_12ST_CORR_SCALE * evalue * exp(logLengthFactor);
        }
        double corrEvalue = pow(evalue, 0.32);
        return corrEvalue;
    }
};


#endif //FOLDSEEK_EVALUENEURALNET_H
