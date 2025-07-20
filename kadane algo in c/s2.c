#include <stdio.h>
#include <limits.h>  // For INT_MIN

int maxSubArray(int* nums, int numsSize) {
    int max_sum = INT_MIN;  // Equivalent to float('-inf')
    int current_sum = 0;

    for (int i = 0; i < numsSize; ++i) {
        current_sum += nums[i];

        if (current_sum > max_sum) {
            max_sum = current_sum;
        }

        if (current_sum < 0) {
            current_sum = 0;
        }
    }

    return max_sum;
}

int main() {
    int nums[] = {-2, 1, -3, 4, -1, 2, 1, -5, 4};
    int size = sizeof(nums) / sizeof(nums[0]);

    int result = maxSubArray(nums, size);
    printf("Max Subarray Sum: %d\n", result);

    return 0;
}
