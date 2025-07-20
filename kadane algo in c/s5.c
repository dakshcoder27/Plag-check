#include <stdio.h>
#include <limits.h>  // For INT_MIN

int max(int a, int b) {
    return (a > b) ? a : b;
}

int maxSubArray(int* nums, int numsSize) {
    int maximumSum = INT_MIN;
    int currSumSubarray = 0;

    for (int i = 0; i < numsSize; ++i) {
        currSumSubarray += nums[i];
        maximumSum = max(maximumSum, currSumSubarray);
        currSumSubarray = max(currSumSubarray, 0);
    }

    return maximumSum;
}

int main() {
    int nums[] = {-2, 1, -3, 4, -1, 2, 1, -5, 4};
    int size = sizeof(nums) / sizeof(nums[0]);

    int result = maxSubArray(nums, size);
    printf("Max Subarray Sum: %d\n", result);

    return 0;
}
