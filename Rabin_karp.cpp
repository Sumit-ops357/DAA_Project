#include <iostream>
#include <string>

using namespace std;

void rabin_karp(const string &text, const string &pattern, int prime) {
    int n = text.length();
    int m = pattern.length();
    int hash_text = 0;
    int hash_pattern = 0;
    int h = 1;
    const int d = 256;

    for (int i = 0; i < m - 1; i++) {
        h = (h * d) % prime;
    }

    for (int i = 0; i < m; i++) {
        hash_pattern = (d * hash_pattern + pattern[i]) % prime;
        hash_text = (d * hash_text + text[i]) % prime;
    }

    for (int i = 0; i <= n - m; i++) {
        if (hash_pattern == hash_text) {
            bool match = true;
            for (int j = 0; j < m; j++) {
                if (text[i + j] != pattern[j]) {
                    match = false;
                    break;
                }
            }
            if (match) {
                cout << "Pattern found at index " << i << endl;
            }
        }

        if (i < n - m) {
            hash_text = (d * (hash_text - text[i] * h) + text[i + m]) % prime;

            if (hash_text < 0) {
                hash_text += prime;
            }
        }
    }
}

int main() {
    string text = "Batch001QualityOKBatch002SpoiledBatch003QualityOK";
    string pattern = "QualityOK";
    int prime = 101;

    cout << "Text: " << text << endl;
    cout << "Pattern: " << pattern << endl;

    rabin_karp(text, pattern, prime);

    return 0;
}