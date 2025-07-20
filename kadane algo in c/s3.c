#include <stdio.h>
#include <limits.h>

int max(int a, int b) {
    return (a > b) ? a : b;
}

int maxSubArray(int* nums, int numsSize) {
    int currSum = nums[0];
    int maxSum = nums[0];

    for (int i = 1; i < numsSize; ++i) {
        currSum = max(nums[i], currSum + nums[i]);  // extend or start new
        maxSum = max(maxSum, currSum);
    }

    return maxSum;
}

int main() {
    int nums[] = {-2, 1, -3, 4, -1, 2, 1, -5, 4};
    int size = sizeof(nums) / sizeof(nums[0]);

    int result = maxSubArray(nums, size);
    printf("Max Subarray Sum: %d\n", result);

    return 0;
}
