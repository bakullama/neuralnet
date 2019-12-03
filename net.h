#ifndef NET_H
#define NET_H
#include <vector>
#include "neuron.h"

using namespace std;
typedef vector<Neuron> Layer;

class Net
{
public:
    Net(const vector<unsigned> &topology);
    void feedForward(const vector<double> &inputVals);
    void backProp(const vector<double> &targetVals);
    void getResults(vector<double> &resultVals) const;

    ~Net();
private:
    vector<Layer> m_layers; // m_layers[layerNum][neuronNum]
    double m_error;
    double m_recentAverageError;
    double m_recentAverageSmoothingFactor;
};

#endif // NET_H
