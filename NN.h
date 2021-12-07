/* To make sure stuff arent included more than once...*/
#pragma once
/*Libraries needed for it to work*/
#include <stdio.h>/*for printf and file io */
#include <vector>/* For C++ fancy dynamic array*/
#include <cstdlib>/* C's standard Library... I don't remember for what */
#include <cmath>/* C's Math library, It is required for defining functions*/
#include <cassert>/* That's for validation of stuff, you can assert facts and if they are not met, the program is terminated*/
#include <iostream>/* like stdio, but for c++ */

struct Connection
{
    double weight;
    double deltaWeight;
};


class Neuron ;
typedef std::vector<Neuron> Layer;

/* *************************** Class Neuron******************************** */

class Neuron
{
public:
    Neuron(unsigned numberOfOutputs, unsigned my_index);
    void setOutputVal(double val)
    {
        outputVal = val;
    }
    double getOutputval(void) const
    {
        return outputVal;
    };
    void feedforward(const Layer &prevlayer);
    void calcOutputGradients(double targetVal);
    void calculateHiddenGradients(const Layer &nextLayer);
    void updateInputWeights(Layer &prevLayer);
private:

    static double eta; //[0.0...1.0] overall training rate
    static double alpha; //[0.0...n] multiplier of last weight change (momentum)

    static double transferFunction(double x);
    static double transferFunctionDerivative(double x);
    static double randomWeight(void)
    {
        return rand() / double(RAND_MAX) ;
    }
    double sumDOW(const Layer &nextLayer) const;
    double outputVal;
    std::vector<struct Connection> outputWeights;
    unsigned mIndex;
    double gradient;

};

/* *************************** Class Net********************************** */
class Net
{

public:

    Net(const std::vector<unsigned> &topology);

    void feedforward(std::vector<double> &input_values);

    void backprop(const std::vector<double> &target_value);

    void getresults(std::vector<double> &resultVals) const;

    double getRecentAverageError();

private:

    std::vector<Layer> layers;
    double mError;
    double recentAverageError;
    double recentAverageSmoothingFactor;
};


