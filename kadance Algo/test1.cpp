#include <iostream>
#include <string>
using namespace std;

void one() {
    int result = 0;
    for (int i = 0; i < 10; i++) {
        result += i * i;
    }
    cout << "Square sum: " << result << endl;

    for (int i = 0; i < 5; i++) {
        cout << i << " ";
    }
    cout << endl;

    result *= 2;
    cout << "Final result: " << result << endl;

    int sum = 0;
    for (int i = 1; i <= 10; i++) {
        sum += i;
    }
    cout << "Sum of 1 to 10: " << sum << endl;
}

void two() {
    int total = 1;
    for (int i = 1; i <= 5; i++) {
        total *= i;
    }
    cout << "Factorial: " << total << endl;

    for (int j = 0; j < 3; j++) {
        cout << "Counting: " << j << endl;
    }

    total += 100;
    cout << "Updated total: " << total << endl;

    int evenSum = 0;
    for (int i = 2; i <= 10; i += 2) {
        evenSum += i;
    }
    cout << "Even sum: " << evenSum << endl;
}

void three() {
    string s = "hello";
    string rev = "";
    for (int i = 0; i < s.length(); i++) {
        rev = s[i] + rev;
    }
    cout << "Reversed string: " << rev << endl;

    int count = 0;
    for (int i = 0; i < 10; i++) {
        if (i % 2 == 0) {
            count++;
        }
    }
    cout << "Even count: " << count << endl;

    int vowels = 0;
    for (char c : s) {
        if (c == 'a' || c == 'e' || c == 'i' || c == 'o' || c == 'u') {
            vowels++;
        }
    }
    cout << "Vowels: " << vowels << endl;
}

int main() {
    one();
    two();
    three();
    return 0;
}
