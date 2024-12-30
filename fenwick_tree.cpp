#include <iostream>
#include <vector>
using namespace std;

class Fenwicktree {
private:
    vector<int> tree;
    int size;

public:
    Fenwicktree(int n) : size(n) {
        tree.resize(n + 1, 0);
    }

    void update(int index, int value) {
        while (index <= size) {
            tree[index] += value;
            index += index & (-index);
        }
    }

    int query(int index) {
        int sum = 0;
        while (index > 0) {
            sum += tree[index];
            index -= index & (-index);
        }
        return sum;
    }

    int range_query(int left, int right) {
        return query(right) - query(left - 1);
    }
};

int main() {
    vector<int> water_levels = {50, 70, 30, 80, 45, 60, 75};
    int n = water_levels.size();

    Fenwicktree fenwick(n);
    for (int i = 0; i < n; i++) {
        fenwick.update(i + 1, water_levels[i]);
    }

    cout << "Total water level in reservoirs 1 to 3: " 
         << fenwick.range_query(1, 3) << " million liters" << endl;

    fenwick.update(3, 10);
    cout << "Updated total water level in reservoirs 1 to 3: " 
         << fenwick.range_query(1, 3) << " million liters" << endl;

    cout << "Total water level in reservoirs 1 to 5: " 
         << fenwick.range_query(1, 7) << " million liters" << endl;

    return 0;
}