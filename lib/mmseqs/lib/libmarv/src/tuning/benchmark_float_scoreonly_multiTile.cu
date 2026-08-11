
#include "benchmarkdata.hpp"
#include "common.hpp"
#include "configtuples.hpp"
#include "benchmark_float_scoreonly.cuh"

#include <vector>

#include "../util.cuh"

std::vector<BenchmarkData> benchmark_float_scoreOnly_multiTile(
    int timingIterations, 
    std::chrono::milliseconds sleepAmount,
    bool checkResults,
    int tiles
){
    using ApproachAndTypeTuple = std::tuple<
        libmarv::ApproachAndType<Approach::unused, Datatype::Float>        
    >;

    assert(tiles > 1);
    constexpr bool singleTile = false;

    using BlockSizeTupleFoo = std::tuple<IC<128>>;
    //using GroupSizeTupleFoo = std::tuple<IC<4>, IC<8>, IC<16>, IC<32>>;
    using GroupSizeTupleFoo = std::tuple<IC<4>>;
    using NumItemsTupleFoo = std::tuple<IC<16>, IC<32>>;
    // using NumItemsTupleFoo = std::tuple<IC<2>>;

    std::vector<BenchmarkData> vec = float_scoreonly::ForEachCombination_call_singleconfigFunction<
        BlockSizeTuple,
        GroupSizeTuple,
        NumItemsTuple,
        // BlockSizeTupleFoo,
        // GroupSizeTupleFoo,
        // NumItemsTupleFoo,
        ApproachAndTypeTuple,
        singleTile
    >{}(timingIterations, sleepAmount, checkResults, tiles);

    return vec;
}