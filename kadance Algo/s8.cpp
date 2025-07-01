#include <iostream>
#include <vector>
using namespace std;
void solve1(){}
int solve(vector<int> nums) {
    int maxSoFar = nums[0], currMax = nums[0];
    // this is being good so far 
    for (int i = 1; i < nums.size(); i++) {
        currMax = max(nums[i], currMax + nums[i]);
        //adding maximum of current element and sum of current maximum with current element
        if (currMax > maxSoFar) maxSoFar = currMax;
    }
    return maxSoFar;
}