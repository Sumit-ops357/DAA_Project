#include <iostream>
#include <vector>
#include <algorithm>

#define MAX 9999

using namespace std;

class dijkstra {
public:
    int dist[100];
    int path[100];
    int visited[100];
    int v;
    int dest;

    void init_path(int cost[50][50]);
    void perform(int cost[50][50]);
    void display_path();
};
void dijkstra::init_path(int cost[50][50]) {
    for (int i = 0; i < v; i++) {
        path[i] = dest;
        dist[i] = cost[i][dest];
        visited[i] = 0;
    }
    visited[dest] = 1;
}
void dijkstra::perform(int cost[50][50]) {
    for (int count = 1; count < v; count++) {
        int u = -1;
        int minDist = MAX;

        for (int i = 0; i < v; i++) {
            if (!visited[i] && dist[i] < minDist) {
                u = i;
                minDist = dist[i];
            }
        }

        if (u == -1) break;

        visited[u] = 1;

        for (int i = 0; i < v; i++) {
            if (!visited[i] && cost[i][u] != MAX) {
                dist[i] = min(dist[i], dist[u] + cost[i][u]);
                if (dist[i] == dist[u] + cost[i][u]) {
                    path[i] = u;
                }
            }
        }
    }
}
void dijkstra::display_path() {
    int src;
    cout << "Enter the source vertex: ";
    cin >> src;

    string locations[] = {"Vertical Farm", "Rooftop Garden", "Backyard Farm", "Urban Greenhouse", "Hydroponic Facility"};

    cout << "The shortest path from " << locations[src-1] << " to " << locations[dest] << " is:" << endl;

    vector<int> spath;
    int curr = src-1;
    while (curr != dest) {
        spath.push_back(curr);
        curr = path[curr];
    }
    spath.push_back(dest);

    for (int i = spath.size() - 1; i >= 0; i--) {
        cout << locations[spath[i]];
        if (i != 0) cout << " -> ";
    }
    cout << endl;

    cout << "Total cost: " << dist[src-1] << endl;
}

int main() {
    int cost[50][50];
    dijkstra d;

    d.v = 5; 

    for (int i = 0; i < d.v; i++) {
        for (int j = 0; j < d.v; j++) {
            cost[i][j] = MAX;
        }
    }

    cost[1][0] = 80;  // Vertical Farm to Rooftop Garden
    cost[2][0] = 20;  // Vertical Farm to Backyard Farm
    cost[3][0] = 100; // Vertical Farm to Urban Greenhouse
    cost[4][0] = 115; // Vertical Farm to Hydroponic Facility
    cost[0][1] = 80;  // Rooftop Garden to Vertical Farm
    cost[4][1] = 90;  // Hydroponic Facility to Rooftop Garden
    cost[0][2] = 20;  // Backyard Farm to Vertical Farm
    cost[3][2] = 60;  // Urban Greenhouse to Backyard Farm
    cost[0][3] = 100; // Urban Greenhouse to Vertical Farm
    cost[2][3] = 60;  // Backyard Farm to Urban Greenhouse
    cost[4][3] = 65;  // Hydroponic Facility to Urban Greenhouse
    cost[0][4] = 115; // Hydroponic Facility to Vertical Farm
    cost[1][4] = 90;  // Rooftop Garden to Hydroponic Facility
    cost[3][4] = 65;  // Urban Greenhouse to Hydroponic Facility

    d.dest = 0; 

    d.init_path(cost);
    d.perform(cost);
    d.display_path();

    return 0;
}