#include "NN.h"


/* *************************** Class Neuron******************************** */
double Neuron::eta = 0.15;//[0.0...1.0] overall training rate
double Neuron::alpha = 0.5;//[0.0...n] multiplier of last weight change (momentum)

Neuron::Neuron(unsigned numberOfOutputs, unsigned my_index)
{
    for (unsigned c = 0; c <= numberOfOutputs ; ++c )
    {
        outputWeights.push_back(Connection());
        outputWeights.back().weight = randomWeight();
    }
    mIndex = my_index;
}

void Neuron::updateInputWeights(Layer &prevLayer)
{
    //weights to be updated are in the connection container
    //in the neurons in the preceding layer
    for (unsigned neuronNumber = 0; neuronNumber < prevLayer.size() ; ++neuronNumber )
    {
        Neuron &neuron = prevLayer[neuronNumber];
        double oldDeltaWeight = neuron.outputWeights[mIndex].deltaWeight;

        double newDeltaWeight =
            //individual input, magnified by the gradient and train rate
            eta
            * neuron.getOutputval()
            * gradient
            //Also add momentum = a fraction of the previous delta weight
            + alpha
            * oldDeltaWeight;

        neuron.outputWeights[mIndex].deltaWeight = newDeltaWeight;
        neuron.outputWeights[mIndex].weight += newDeltaWeight;


    }
}

double Neuron::sumDOW(const Layer &nextLayer) const
{
    double sum = 0.0;
    // sum of  contributions of the errors at the modes we feed
    for(unsigned neuronNumber = 0; neuronNumber < nextLayer.size() -1; neuronNumber++)
    {
        sum += outputWeights[neuronNumber].weight * nextLayer[neuronNumber].gradient;
    }
}

void Neuron::calculateHiddenGradients(const Layer &nextLayer)
{
    double dow = sumDOW(nextLayer);
    gradient = dow * Neuron::transferFunctionDerivative(outputVal);
}

void Neuron::calcOutputGradients(double targetVal)
{
    double delta = targetVal - outputVal;
    gradient = delta * Neuron::transferFunctionDerivative(outputVal);
}

double Neuron::transferFunction(double x)
{
    return tanh(x);
}

double Neuron::transferFunctionDerivative(double x)
{
    return 1.0 - (x * x);
}

void Neuron::feedforward(const Layer &prevlayer)
{
    double sum = 0.0;

    for (unsigned neuronNumber = 0; neuronNumber < prevlayer.size(); ++neuronNumber)
    {
        sum += prevlayer[neuronNumber].outputVal;
        prevlayer[neuronNumber].outputWeights[mIndex].weight;
    }

    outputVal = Neuron::transferFunction(sum);

}

/* *************************** Class Net********************************** */

Net::Net(const std::vector<unsigned> &topology)
{
    unsigned numLayers = topology.size();

    for (unsigned layerNum = 0; layerNum < numLayers ; layerNum++ )
    {
        layers.push_back(Layer());
        unsigned numberOfOutputs = layerNum == topology.size() - 1 ? 0: topology[layerNum + 1];

        for (unsigned neuronNumber = 0; neuronNumber <= topology[layerNum] ; ++neuronNumber )
        {
            printf("Made a Neuron! \n");
            layers.back().push_back(Neuron(numberOfOutputs, neuronNumber));
        }

    }
    //force the bias node's output to 1.0. it is the last neuron created above
    layers.back().back().setOutputVal(1.0);
}

void Net::getresults(std::vector<double> &resultVals) const
{
    resultVals.clear();
    for(unsigned neuronNumber = 0; neuronNumber < layers.back().size() - 1; ++neuronNumber)
    {
        resultVals.push_back(layers.back()[neuronNumber].getOutputval());
    }
}

void Net::feedforward(std::vector<double> &input_values)
{
    assert(input_values.size() == layers[0].size() - 1);
    for (unsigned i = 0; i < input_values.size(); i++ )
    {
        layers[0][i].setOutputVal(input_values[i]);
    }

    for (unsigned layerNum = 1; layerNum < layers.size() ; layerNum++ )
    {
        Layer &prevLayer = layers[layerNum - 1];
        for (unsigned neuronNum = 0; neuronNum < layers[layerNum].size() ; neuronNum++ )
        {
            layers[layerNum][neuronNum].feedforward(prevLayer);
        }
    }

}

void Net::backprop(const std::vector<double> &target_value)
{
    //calcualte overall net error (RMS of output neuron errors)
    Layer &outputlayer = layers.back();
    mError = 0.0;
    for(unsigned neuronNumber = 0; neuronNumber < outputlayer.size() -1; ++neuronNumber)
    {
        double delta = target_value[neuronNumber] - outputlayer[neuronNumber].getOutputval();
        mError += delta * delta;
    }
    mError /= outputlayer.size() -1; //get average error squared
    mError = sqrt(mError);//RMS

    //implement a recent average measurement

    recentAverageError =
        (recentAverageError * recentAverageSmoothingFactor + mError)
        / (recentAverageSmoothingFactor + 1.0);


    //calculate output layer gradient

    for (unsigned neuronNumber = 0; neuronNumber < outputlayer.size() -1 ; ++neuronNumber)
    {
        outputlayer[neuronNumber].calcOutputGradients(target_value[neuronNumber]);
    }


    //calculate gradients on hidden layers
    for (unsigned layerNumber = layers.size() - 2; layerNumber > 0 ; --layerNumber)
    {
        Layer &hiddenLayer = layers[layerNumber];
        Layer &nextLayer = layers[layerNumber + 1];

        for(unsigned neuronNumber = 0; neuronNumber < hiddenLayer.size(); neuronNumber++)
        {
            hiddenLayer[neuronNumber].calculateHiddenGradients(nextLayer);
        }
    }

    //for all layers from output layer to frist hidden layer
    //update connection weights
    for(unsigned layerNumber = layers.size() - 1; layerNumber > 0; --layerNumber)
    {
        Layer &layer = layers[layerNumber];
        Layer &prevLayer = layers[layerNumber - 1];
        for(unsigned neuronNumber = 0; neuronNumber < layer.size() - 1; ++neuronNumber)
        {
            layer[neuronNumber].updateInputWeights(prevLayer);
        }
    }


}

double Net::getRecentAverageError()
{
    return recentAverageError;
}
