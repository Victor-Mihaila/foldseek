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
    static constexpr double WINDOWED_EVALUE_CORR_EXPONENT = 0.8559494682342873;
    static constexpr double WINDOWED_EVALUE_CORR_SCALE = 0.13415799519899208;
    static constexpr double LEGACY_EVALUE_CORR_EXPONENT = 0.32;
    // mode 2: E = scale * N * sqrt(p_q * p_t) * exp(f(u) - f(u_ref)), f(u) = c1 u + c2 u^2, u = log of the shorter of query and
    // target length clipped to [min, max], u_ref = log(ref); scale is the multiplier when the shorter protein has the reference
    // length. p_q, p_t come from the query's and the target's own mu/lambda (p_q alone if the target has none)
    static constexpr double LEGACY_12ST_EVALUE_CORR_SCALE = 0.1357792528025477;
    static constexpr double LEGACY_12ST_LENGTH_REF = 150.0;
    static constexpr double LEGACY_12ST_LENGTH_C1 = -1.8318504160819078;
    static constexpr double LEGACY_12ST_LENGTH_C2 = 0.45780993271377185;
    static constexpr double LEGACY_12ST_LENGTH_MIN = 60.0;
    static constexpr double LEGACY_12ST_LENGTH_MAX = 400.0;
    static constexpr double WINDOWED_PRED_SLOPE = 0.5142189860343933;
    static constexpr double WINDOWED_PRED_INTERCEPT = 35.95916748046875;
    static constexpr double WINDOWED_MIN_LAMBDA = 9.999999747378752e-05;
    static constexpr float WINDOWED_PROFILE_BIT_FACTOR = 8.0f;
    static constexpr int LEGACY_12ST_FEATURE_SIZE = 35;
    static constexpr float LEGACY_12ST_PROFILE_BIT_FACTOR = 8.0f;
    static constexpr double LEGACY_12ST_LAMBDA_MEAN = 0.05809067;
    static constexpr double LEGACY_12ST_LAMBDA_STD = 0.03113629892;
    static constexpr double LEGACY_12ST_MU_MEAN = 7.340611654;
    static constexpr double LEGACY_12ST_MU_STD = 7.853373165;
    BaseMatrix *subMat;
    double logDbResidueCount;
    double logDbSequenceCount;
    int mode;
    bool recoverProfileComposition;
    const BaseMatrix *subMat12St;
    KerasModel encoder;
    Tensor in;
    Tensor out;
    KerasModel windowedEncoder;
    Tensor windowedIn;
    Tensor windowedOut;

    std::vector<int> stateToWindowedIndex;
    std::vector<float> windowedBackground;

    static float softplus(float x);
    int predictedAlignmentLength(double score) const;
    void fillLegacy12StComposition(float *dst, const Sequence &seq, const char *alphabet, int alphabetSize, const BaseMatrix *backgroundMat) const;
    std::pair<double, double> predictMuLambdaLegacy12St(const Sequence &seq3Di, const Sequence &seq12St);
    void fillWindowedComposition(float *dst, const unsigned char *seq, unsigned int seqLen, int end1Based, int spanLen) const;
    void fillWindowedComposition(float *dst, const Sequence &seq, int end1Based, int spanLen) const;
    std::pair<double, double> predictMuLambdaWindowed(const unsigned char *querySeq,
                                                      unsigned int queryLen,
                                                      int queryEnd1Based,
                                                      const unsigned char *targetSeq,
                                                      unsigned int targetLen,
                                                      int targetEnd1Based,
                                                      double score);
    std::pair<double, double> predictMuLambdaWindowed(const Sequence &querySeq,
                                                      int queryEnd1Based,
                                                      const Sequence &targetSeq,
                                                      int targetEnd1Based,
                                                      double score);
public:

    EvalueNeuralNet(size_t dbResidueCount,
                    size_t dbSequenceCount,
                    BaseMatrix* subMat,
                    int mode = LocalParameters::EVALUE_NN_MODE_LEGACY,
                    bool legacyUses12St = false,
                    bool recoverProfileComposition = false,
                    const BaseMatrix *subMat12St = NULL);

    std::pair<double, double> predictMuLambda(const unsigned char * seq, unsigned int L);
    std::pair<double, double> predictMuLambda(const Sequence &seq);
    std::pair<double, double> predictMuLambda(const Sequence &seq3Di, const Sequence &seq12St);
    std::pair<double, double> predictMuLambda(const unsigned char *querySeq,
                                              unsigned int queryLen,
                                              int queryEnd1Based,
                                              const unsigned char *targetSeq,
                                              unsigned int targetLen,
                                              int targetEnd1Based,
                                              double score);
    std::pair<double, double> predictMuLambda(const Sequence &querySeq,
                                              int queryEnd1Based,
                                              const Sequence &targetSeq,
                                              int targetEnd1Based,
                                              double score);
    bool usesWindowedModel() const { return mode == LocalParameters::EVALUE_NN_MODE_WINDOW; }
    bool usesLegacy12StModel() const { return mode == LocalParameters::EVALUE_NN_MODE_LEGACY_12ST; }
    double activeLogDbCount() const { return (usesWindowedModel() || usesLegacy12StModel()) ? logDbSequenceCount : logDbResidueCount; }

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
        if (usesLegacy12StModel() && targetMuLambda != NULL) {
            logPVal = 0.5 * (logPVal + computePvalue(score, targetMuLambda->first, targetMuLambda->second));
        }
	double dbSizeTimesLogPVal = logPVal + activeLogDbCount();
	double evalue = exp(dbSizeTimesLogPVal);
        if (usesWindowedModel()) {
            return WINDOWED_EVALUE_CORR_SCALE * pow(evalue, WINDOWED_EVALUE_CORR_EXPONENT);
        }
        if (usesLegacy12StModel()) {
            double u = log(std::min(std::max(static_cast<double>(std::min(queryLen, targetLen)), LEGACY_12ST_LENGTH_MIN), LEGACY_12ST_LENGTH_MAX));
            double uRef = log(LEGACY_12ST_LENGTH_REF);
            double logLengthFactor = (LEGACY_12ST_LENGTH_C2 * u + LEGACY_12ST_LENGTH_C1) * u
                                   - (LEGACY_12ST_LENGTH_C2 * uRef + LEGACY_12ST_LENGTH_C1) * uRef;
            return LEGACY_12ST_EVALUE_CORR_SCALE * evalue * exp(logLengthFactor);
        }
	double corrEvalue = pow(evalue, LEGACY_EVALUE_CORR_EXPONENT);
        return corrEvalue;
    }
};


#endif //FOLDSEEK_EVALUENEURALNET_H
