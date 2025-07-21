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
int dynamicProgramming(int* nums, int numsSize) {
    int max_sum = INT_MIN;
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
int divideAndConquer(int* nums, int left, int right) {
    if (left > right) return INT_MIN;

    if (left == right) return nums[left];

    int mid = (left + right) / 2;

    int left_max = divideAndConquer(nums, left, mid - 1);
    int right_max = divideAndConquer(nums, mid + 1, right);

    int cross_max = nums[mid];
    int temp_sum = 0;

    for (int i = mid - 1; i >= left; --i) {
        temp_sum += nums[i];
        if (temp_sum > cross_max) {
            cross_max = temp_sum;
        }
    }

    temp_sum = 0;
    for (int i = mid + 1; i <= right; ++i) {
        temp_sum += nums[i];
        if (temp_sum > cross_max) {
            cross_max += temp_sum;
        }
    }

    return max(max(left_max, right_max), cross_max);
}