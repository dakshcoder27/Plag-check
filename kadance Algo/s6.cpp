#include <iostream>
#include <vector>
using namespace std;

int maximumSubArraySum(vector<int> arr) {
    int maxTillNow = arr[0], maxTotal = arr[0];
    for (int i = 1; i < arr.size(); i++) {
        maxTillNow = max(arr[i], maxTillNow + arr[i]);
        maxTotal = max(maxTotal, maxTillNow);
    }
    return maxTotal;
}
