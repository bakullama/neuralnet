#include "net.h"
#include <vector>
#include <iostream>
#include <cmath>


int main() {
    std::vector<unsigned> topology;
    topology.push_back(2); // input
    topology.push_back(5); // hidden
    topology.push_back(1); // output

    Net net(topology);

    const std::vector<double> inputVals[4] = {
        {0, 0}, {0, 1}, {1, 0}, {1, 1}
    };
    const std::vector<double> targetVals[4] = {
        {0}, {1}, {1}, {0}
    };


    for (int i = 0; i < 1000; ++i) {
        for (int j = 0; j < 4; ++j) {
            net.feedForward(inputVals[j]); // training
            net.backProp(targetVals[j]);
        }
    }


    std::vector<double> resultVals;
    for (int j = 0; j < 4; ++j) {
        std::cout << "input: | " << inputVals[j][0] << " | " << inputVals[j][1];
        std::cout << " | :: output: ";
        net.feedForward(inputVals[j]);
        net.getResults(resultVals);
        std::cout << resultVals[0] << std::endl;
    }



    return 0;
}
