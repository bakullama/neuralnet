#include "net.h"
#include "neuron.h"
#include <vector>
#include <iostream>
#include <cassert>
#include <cmath>

using namespace std;

Net::~Net() {
    delete(&m_layers);
    delete(&m_error);
    delete(&m_recentAverageError);
    delete(&m_recentAverageSmoothingFactor);
}

Net::Net(const vector<unsigned> &topology)
{


    unsigned long numLayers = topology.size(); // 3
    for (unsigned layerNum = 0; layerNum < numLayers; ++layerNum) {

        m_layers.push_back(Layer()); // create new empty layer and append to m_layers
        unsigned numOutputs = layerNum == topology.size() - 1 ? 0 : topology[layerNum + 1];
        cout << "new layer " << layerNum << endl;
        // fillin new layer with neurons
        for (unsigned neuronNum = 0; neuronNum <= topology[layerNum]; ++neuronNum) { // <= means 1 extra neuron to act as bias for the layer
            m_layers.back().push_back(Neuron(numOutputs, neuronNum));
            cout << "new neuron" << endl;
        }
    }

    // force bias to 1.0
    m_layers.back().back().setOutputVal(1.0);
}

void Net::getResults(vector<double> &resultVals) const {
    resultVals.clear();
    for (unsigned neuronNum = 0; neuronNum < m_layers.back().size() - 1; ++neuronNum) {
        resultVals.push_back(m_layers.back()[neuronNum].getOutputVal());
    }
}

void Net::feedForward(const vector<double> &inputVals) {

    assert(inputVals.size() == m_layers[0].size() - 1);
    // assign input values to input neurons
    for (unsigned inputNum = 0; inputNum < inputVals.size(); ++inputNum) {
        m_layers[0][inputNum].setOutputVal(inputVals[inputNum]);
    }

    // forward propagate
    for (unsigned layerNum = 1; layerNum < m_layers.size(); ++layerNum) { // starting at 1 as input layer already set
        Layer &prevLayer = m_layers[layerNum - 1]; // pointer to previous layer
        for (unsigned neuronNum = 0; neuronNum < m_layers[layerNum].size() - 1; ++neuronNum) {
            m_layers[layerNum][neuronNum].feedForward(prevLayer);
        }
    }
}

void Net::backProp(const vector<double> &targetVals) {
    // calculate overall net error (Root Mean Square error of output neuronNum errors)
    Layer &outputLayer = m_layers.back();
    m_error = 0.0;

    for (unsigned neuronNum = 0; neuronNum < outputLayer.size() - 1; ++neuronNum) {
        double delta = targetVals[neuronNum] - outputLayer[neuronNum].getOutputVal();
        m_error += delta * delta;
    }
    m_error /= outputLayer.size() - 1; // get avg
    m_error = sqrt(m_error);

    m_recentAverageError =
            (m_recentAverageError - m_recentAverageSmoothingFactor + m_error)
            / (m_recentAverageSmoothingFactor + 1.0);

    // calculate output layer gradients

    for (unsigned neuronNum = 0; neuronNum < outputLayer.size() - 1; ++neuronNum) {
        outputLayer[neuronNum].calcOutputGradients(targetVals[neuronNum]);
    }

    // calcualate hidden layers gradients

    for (unsigned layerNum = m_layers.size() - 2; layerNum > 0; --layerNum) { // - 2 for rightmost hidden layer
        Layer &hiddenLayer = m_layers[layerNum];
        Layer &nextLayer = m_layers[layerNum + 1];
        for (unsigned neuronNum = 0; neuronNum < hiddenLayer.size() - 1; ++neuronNum ) {
            hiddenLayer[neuronNum].calcHiddenGradients(nextLayer);
        }

    }

    // for all layers from output to 1st hidden layer

    // update connection weights
    for (unsigned layerNum = m_layers.size() - 1; layerNum > 0; --layerNum) {
        Layer &layer = m_layers[layerNum];
        Layer &prevLayer = m_layers[layerNum - 1];

        for (unsigned neuronNum = 0; neuronNum < layer.size(); ++neuronNum) {
            layer[neuronNum].updateInputWeights(prevLayer);
        }
    }

}

