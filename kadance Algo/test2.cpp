#include <iostream>
#include <string>
using namespace std;

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

void one() {
    int result = 0;
    for (int i = 0; i < 10; i++) {
        result += i * i;
    }
    cout << "Square sum: " << result << endl;

    for (int i = 0; i < 5; i++) {
        cout << i << " ";
    }
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

int main() {
    three();
    one();
    two();
    return 0;
}
