#include <iostream>
#include <vector>
using namespace std;

int maxSubArray(vector<int>& arr) {
    int localMax = arr[0];
    int globalMax = arr[0];
    for (size_t i = 1; i < arr.size(); i++) {
        localMax = (arr[i] > localMax + arr[i]) ? arr[i] : localMax + arr[i];
        if (localMax > globalMax) globalMax = localMax;
    }
    return globalMax;
}