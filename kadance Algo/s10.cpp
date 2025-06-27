int maxSubSum(vector<int>& nums) {
    int sum = nums[0];
    int maxAns = nums[0];
    for (int i = 1; i < nums.size(); i++) {
        sum = (sum > 0) ? sum + nums[i] : nums[i];
        maxAns = max(maxAns, sum);
    }
    return maxAns;
}
