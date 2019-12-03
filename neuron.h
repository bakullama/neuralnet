#ifndef NEURON_H
#define NEURON_H
#include <vector>
#include "connection.h"

using namespace std;


class Neuron
{
    typedef vector<Neuron> Layer;

public:
    Neuron(unsigned numOutputs, unsigned myIndex);
    void feedForward(const Layer &prevLayer);

    void setOutputVal(double val);
    double getOutputVal(void) const;

    void calcOutputGradients(double targetVal);
    void calcHiddenGradients(const Layer& nextLayer);

    void updateInputWeights(Layer& prevLayer);


private:
    static double randomWeight();

    static double activationFunction(double x);
    static double activationFunctionDerivative(double x);

    double sumDOW(const Layer& nextLayer) const;

    double m_outputVal;
    unsigned m_myIndex;
    double m_gradient;
    vector<Connection> m_outputWeights;

    static double lr;

};

#endif // NEURON_H
