//make sure stuff are included only once
#pragma once
//includes
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>

class TrainingData
{
    public:
        TrainingData(const std::string path);
        void getTopology(std::vector<unsigned> &topology);
        int isEof(void){return trainingDataFile.eof();}
        unsigned getNextInputs(std::vector <double> &inputVals);
        unsigned getTargetOutputs(std::vector <double> &targetVals);

    private:
        std::ifstream trainingDataFile;

};

/* File Format:

    topology: <no. of neurons per layer, separated by one space>
    in: <inputs, separated by space>
    out: <target outputs, separated by space>
    .
    .
    .

*/

