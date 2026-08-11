

#ifndef BENCHMARKDATA_HPP
#define BENCHMARKDATA_HPP

#include "common.hpp"

#include <string>
#include <vector>
#include <algorithm>
#include <iostream>




struct BenchmarkData{
    int blocksize;
    int tilesize;
    int groupsize;
    int numItems;
    float mingcups;
    float avggcups;
    float maxgcups;
    std::string datatypeString;
    std::string approachString;
    int numTiles;
};


std::ostream& operator<<(std::ostream& os, const BenchmarkData& data);

void printBenchmarkData_full(std::vector<BenchmarkData> vec, char sep = ',');
void printBenchmarkData_bestPerTile(std::vector<BenchmarkData> vec, char sep = ',');

void printBestAsConfig(std::vector<BenchmarkData> vec);

template<
    int blocksize,
    int groupsize,
    int numItems,
    int tilesize,
    class ApproachAndType
>
BenchmarkData makeNotExecutedBenchmarkData(){
    BenchmarkData benchmarkData;
    benchmarkData.blocksize = blocksize;
    benchmarkData.tilesize = tilesize;
    benchmarkData.groupsize = groupsize;
    benchmarkData.numItems = numItems;
    benchmarkData.mingcups = 0;
    benchmarkData.avggcups = 0;
    benchmarkData.maxgcups = 0;
    benchmarkData.datatypeString = to_string(ApproachAndType::datatype);
    benchmarkData.approachString = to_string(ApproachAndType::approach);
    benchmarkData.numTiles = 0;

    return benchmarkData;
}


#endif