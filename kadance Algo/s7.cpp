#include <iostream>
#include <vector>
using namespace std;

int solve(vector<int> nums) {
    int maxSoFar = nums[0], currMax = nums[0];
    for (int i = 1; i < nums.size(); i++) {
        currMax = max(nums[i], currMax + nums[i]);
        if (currMax > maxSoFar) maxSoFar = currMax;
    }
    return maxSoFar;
}
