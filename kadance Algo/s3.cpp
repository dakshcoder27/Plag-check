#include <iostream>
#include <vector>
using namespace std;

int kadaneAlgo(vector<int> nums) {
    int best = INT8_MIN, sum = 0;
    for (int val : nums) {
        sum = max(val, sum + val);
        best = max(best, sum);
    }
    return best;
}