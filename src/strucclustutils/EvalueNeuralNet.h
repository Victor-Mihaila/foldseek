//
// Created by Martin Steinegger on 19/01/2022.
//

#ifndef FOLDSEEK_EVALUENEURALNET_H
#define FOLDSEEK_EVALUENEURALNET_H
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
    static constexpr double LEGACY_12ST_EVALUE_CORR_EXPONENT = 1.0;
    static constexpr double LEGACY_12ST_EVALUE_CORR_SCALE = 0.25;
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
    void fillLegacy12StComposition(float *dst, const Sequence &seq, const char *alphabet, int alphabetSize) const;
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
                    bool recoverProfileComposition = false);

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
    
    double computeEvalueCorr(double score, double lambda_, double mu){
	double logPVal = computePvalue(score, lambda_, mu);    
	double dbSizeTimesLogPVal = logPVal + activeLogDbCount();
	double evalue = exp(dbSizeTimesLogPVal);
        if (usesWindowedModel()) {
            return WINDOWED_EVALUE_CORR_SCALE * pow(evalue, WINDOWED_EVALUE_CORR_EXPONENT);
        }
        if (usesLegacy12StModel()) {
            return LEGACY_12ST_EVALUE_CORR_SCALE * pow(evalue, LEGACY_12ST_EVALUE_CORR_EXPONENT);
        }
	double corrEvalue = pow(evalue, LEGACY_EVALUE_CORR_EXPONENT);
        return corrEvalue;
    }
};


#endif //FOLDSEEK_EVALUENEURALNET_H
