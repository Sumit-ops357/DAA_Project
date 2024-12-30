#include <iostream>
#include <vector>
#include <string>

using namespace std;

struct Crop {
    string name;
    int demand;
};

int Partition(vector<Crop>& crops, int l, int r) {
    int pivot = crops[l].demand;
    int i = l;
    int j = r + 1;

    while (true) {
        do {
            i++;
        } while (i <= r && crops[i].demand > pivot);

        do {
            j--;
        } while (crops[j].demand < pivot);

        if (i >= j) {
            break;
        }

        swap(crops[i], crops[j]);
    }

    swap(crops[l], crops[j]);

    return j;
}

void QuickSortCrops(vector<Crop>& crops, int l, int r) {
    if (l < r) {
        int s = Partition(crops, l, r);
        QuickSortCrops(crops, l, s - 1);
        QuickSortCrops(crops, s + 1, r);
    }
}

int main() {
    vector<Crop> crops = {
        {"Tomatoes", 500},
        {"Lettuce", 300},
        {"Carrots", 700},
        {"Spinach", 150},
        {"Peppers", 400},
        {"Cucumbers", 600},
        {"Broccoli", 400},
        {"Eggplants", 50}
    };

    cout << "Original Crop Demands: " << endl;
    for (const auto& crop : crops) {
        cout << crop.name << ": " << crop.demand << endl;
    }
    cout << endl;

    QuickSortCrops(crops, 0, crops.size() - 1);

    cout << "Sorted Crop Demands according Demand: " << endl;
    for (const auto& crop : crops) {
        cout << crop.name << ": " << crop.demand << endl;
    }
    cout << endl;

    return 0;
}