#include <stdio.h>
#include <stdlib.h>

int max(int a, int b) {
    return (a > b) ? a : b;
}

int maxSubArray(int* nums, int numsSize) {
    // Create a DP array (like Python's arr[])
    int* arr = (int*)malloc(numsSize * sizeof(int));
    arr[0] = nums[0];

    int maxSum = arr[0];

    for (int i = 1; i < numsSize; ++i) {
        arr[i] = max(nums[i], arr[i - 1] + nums[i]);

        if (arr[i] > maxSum) {
            maxSum = arr[i];
        }
    }

    free(arr);  // Always free dynamically allocated memory
    return maxSum;
}

int main() {
    int nums[] = {-2, 1, -3, 4, -1, 2, 1, -5, 4};
    int size = sizeof(nums) / sizeof(nums[0]);

    int result = maxSubArray(nums, size);
    printf("Max Subarray Sum: %d\n", result);

    return 0;
}
