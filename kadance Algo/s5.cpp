#include <iostream>
#include <vector>
using namespace std;

int maxSum(vector<int>& nums) {
    int curr = nums[0];
    int result = nums[0];
    for (int i = 1; i < nums.size(); ++i) {
        if (curr < 0) curr = 0;
        curr += nums[i];
        result = max(result, curr);
    }
    return result;
}