#include <iostream>
#include <vector>
#include <string>

using namespace std;

struct Edge {
    int u, v, weight;
};

void unionSets(vector<int> &arr, int n, int u, int v) {
    int temp = arr[u];
    for (int i = 0; i < n; i++) {
        if (arr[i] == temp) {
            arr[i] = arr[v];
        }
    }
}

int findSet(vector<int> &arr, int u, int v) {
    return arr[u] == arr[v];
}

void heapify(vector<Edge> &A, int n, int i) {
    int largest = i;
    int left = 2 * i + 1;
    int right = 2 * i + 2;

    if (left < n && A[left].weight > A[largest].weight) {
        largest = left;
    }

    if (right < n && A[right].weight > A[largest].weight) {
        largest = right;
    }

    if (largest != i) {
        swap(A[i], A[largest]);
        heapify(A, n, largest);
    }
}

void heapSort(vector<Edge> &A) {
    int n = A.size();

    for (int i = n / 2 - 1; i >= 0; i--) {
        heapify(A, n, i);
    }

    for (int i = n - 1; i > 0; i--) {
        swap(A[0], A[i]);
        heapify(A, i, 0);
    }
}

vector<Edge> kruskalMST(vector<Edge> &edges, vector<int> &arr, int n) {
    vector<Edge> mst;
    int n1 = edges.size();
    for (int i = 0; i < n1; i++) {
        int u = edges[i].u;
        int v = edges[i].v;
        if (!findSet(arr, u, v)) {
            mst.push_back(edges[i]);
            unionSets(arr, n, u, v);
        }
    }
    return mst;
}

int main() {
    vector<string> locations = {"Water Source 0", "Reservoir 1", "Reservoir 2", "Reservoir 3", "Reservoir 4", "Reservoir 5", "Reservoir 6", "Reservoir 7"};
    vector<vector<int>> graph = {
        {0, 105, 65, 80, 0, 0, 0, 0},
        {105, 0, 45, 0, 0, 40, 0, 100},
        {65, 45, 0, 50, 60, 30, 0, 0},
        {80, 0, 50, 0, 40, 28, 0, 0},
        {0, 0, 60, 40, 0, 0, 70, 0},
        {0, 35, 30, 0, 0, 0, 55, 40},
        {0, 0, 0, 0, 70, 55, 0, 90},
        {0, 100, 0, 0, 0, 40, 90, 0},
    };

    vector<Edge> edges;
    int n = graph.size();

    vector<int> arr(n);
    for (int i = 0; i < n; i++) {
        arr[i] = i;
    }

    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            if (graph[i][j] != 0) {
                edges.push_back(Edge{i, j, graph[i][j]});
            }
        }
    }

    heapSort(edges);

    vector<Edge> mst = kruskalMST(edges, arr, n);

    cout << "\nEdges and their weights:\n";
    int n2 = edges.size();
    for (int i = 0; i < n2; i++) {
        cout << "(" << locations[edges[i].u] << ", " << locations[edges[i].v] << ") --> " << edges[i].weight << "\n";
    }

    cout << "\nEdges in the MST:\n";
    int m = mst.size();
    int totalCost = 0;
    for (int i = 0; i < m; i++) {
        cout << "(" << locations[mst[i].u] << ", " << locations[mst[i].v] << ") --> " << mst[i].weight << "\n";
        totalCost += mst[i].weight;
    }

    cout << "\nMinimum Cost of the MST: " << totalCost << "\n";

    return 0;
}