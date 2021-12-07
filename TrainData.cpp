#include "TrainData.h"


TrainingData::TrainingData(const std::string path)
{
    trainingDataFile.open(path.c_str());
}

unsigned TrainingData::getNextInputs(std::vector <double> &inputVals)
{
    inputVals.clear();

    std::string line;
    getline(trainingDataFile, line);
    std::stringstream ss(line);

    std::string label;
    ss >> label;

    if(label.compare("in:") == 0)
    {
        double onevalue;
        while(ss >> onevalue)
        {
            inputVals.push_back(onevalue);
        }
    }

    return inputVals.size();


}

unsigned TrainingData::getTargetOutputs(std::vector <double> &targetVals)
{
    targetVals.clear();

    std::string line;
    getline(trainingDataFile, line);
    std::stringstream ss(line);

    std::string label;
    ss >> label;

    if(label.compare("out:") == 0)
    {
        double onevalue;
        while(ss >> onevalue)
        {
            targetVals.push_back(onevalue);
        }
    }

    return targetVals.size();
}

void TrainingData::getTopology(std::vector<unsigned> &topology)
{
    std::string line;
    std::string label;

    std::getline(trainingDataFile, line);
    std::stringstream ss(line);

    ss >> label;

    if(this->isEof() || label.compare("topology:") != 0)
    {
        abort();
    }

    while(!ss.eof())
    {
        unsigned n;
        ss>> n;
        topology.push_back(n);
    }

    return ;
}


