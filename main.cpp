#include "net.h"
#include <vector>
#include <iostream>

using namespace std;

int main() {
    vector<unsigned> topology;
    topology.push_back(2); // input
    topology.push_back(5); // hidden
    topology.push_back(1); // output
    Net net(topology);

    const vector<double> inputVals[4] = {
        {0, 0}, {0, 1}, {1, 0}, {1, 1}
    };
    const vector<double> targetVals[4] = {
        {0}, {1}, {1}, {0}
    };
    cout << inputVals[0][0] << endl;

    for (int i = 0; i < 1000; ++i) {
        cout << "start generation: " << i << endl;
        for (int j = 0; j < 4; ++j) {
            cout << j << endl;
            net.feedForward(inputVals[j]); // training
            net.backProp(targetVals[j]);
        }
        cout << "end generation" << endl;
    }

    net.feedForward(inputVals[0]);
    vector<double> resultVals;
    for (int j = 0; j < 4; ++j) {
        cout << "input: " << inputVals[j][0] << ", " << inputVals[j][1] << endl;
        cout << "output: ";
        net.feedForward(inputVals[j]);
        net.getResults(resultVals);
        cout << resultVals[0] << endl;
    }
    cout << "end" << endl;
    return 0;
}
