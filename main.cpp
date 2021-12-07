#include "NN.h"
#include "TrainData.h"

/* **********************************Functions********************************* */

void showVectorVals(std::string label, std::vector<double> &v)
{
    std::cout << label ;
    for (int i = 0;i < v.size() ; i++ )
    {
        std::cout << v[i] << " " ;
    }
    std::cout << std::endl;
}

/* **********************************Main********************************* */
int main(int argc, char** argv)
{
    printf("Hello, Universe! \n");
    TrainingData  trainingData("C:\\Users\\Admin\\Documents\\C_PROJECTS\\NNCpp\\trainer.txt");

    std::vector<unsigned> topology;
    trainingData.getTopology(topology);
    Net myNet(topology);

    std::vector<double> inputvals, targetvals, resultvals;
    int trainingpass = 0;

    while (!trainingData.isEof())
    {
        ++ trainingpass;
        std::cout << std::endl << "pass: " << trainingpass << std::endl;

        //Get new input data and feed it forward

        if(trainingData.getNextInputs(inputvals) != topology[0])
        {
            printf("Error");
            break;
        }
        showVectorVals(": Inputs: ", inputvals);
        myNet.feedforward(inputvals);

        //collect the net's actual results
        myNet.getresults(resultvals);
        showVectorVals(": Outputs:", resultvals);

        //Train the net what the outputs should have been
        trainingData.getTargetOutputs(targetvals);
        showVectorVals("Targets:", targetvals);
        assert(targetvals.size() == topology.back());

        myNet.backprop(targetvals);

        // Report how well the training is working
        std::cout << "Net recent average error: " << myNet.getRecentAverageError() << std::endl ;

    }

    std::cout << std::endl << " Training Done" ;

    return 0;
}
