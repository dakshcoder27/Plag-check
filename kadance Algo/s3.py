class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        currSum = maxSum = nums[0]

        for i in range(1, len(nums)):
            currSum = max(nums[i], currSum + nums[i])  # extend or start new
            maxSum = max(maxSum, currSum)

        return maxSum