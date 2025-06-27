#include <iostream>
#include <vector>
using namespace std;

int kadane(vector<int> nums) {
    int currentMax = nums[0], overallMax = nums[0];
    for (int i = 1; i < nums.size(); i++) {
        currentMax = max(nums[i], currentMax + nums[i]);
        overallMax = max(overallMax, currentMax);
    }
    return overallMax;
}