#include <iostream>
#include <vector>
using namespace std;

int findMaxSubArraySum(vector<int>& nums) {
    int maxEndingHere = 0, maxSoFar = INT8_MIN;
    for (auto num : nums) {
        maxEndingHere += num;
        maxSoFar = max(maxSoFar, maxEndingHere);
        if (maxEndingHere < 0) maxEndingHere = 0;
    }
    return maxSoFar;
}