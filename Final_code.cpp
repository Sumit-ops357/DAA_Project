#include<iostream> 
#include<vector>
#include<algorithm>
#include<string>
#include<climits>
#include<queue>
#include<unordered_map>
#include<set>
using namespace std;

int minDistance(const vector<int>& dist, const vector<bool>& sptSet, int V) {
    int min = INT_MAX, min_index;
    for (int v = 0; v < V; v++) {
        if (!sptSet[v] && dist[v] <= min) {
            min = dist[v];
            min_index = v;
        }
    }
    return min_index;
}

// Dijkstra's algorithm to find the shortest path in the graph
void shortestPathDijkstra(const vector<vector<int>>& graph, int src) {
    int V = graph.size();  // Number of vertices
    vector<int> dist(V, INT_MAX);  // Array to store the shortest distance from source
    vector<bool> sptSet(V, false);  // Boolean array to track the shortest path tree

    dist[src] = 0;  // Distance from source to itself is 0

    // Find the shortest path for all vertices
    for (int count = 0; count < V - 1; count++) {
        int u = minDistance(dist, sptSet, V);  // Get the vertex with the minimum distance
        sptSet[u] = true;

        // Update the distance of the adjacent vertices of the selected vertex
        for (int v = 0; v < V; v++) {
            if (!sptSet[v] && graph[u][v] && dist[u] != INT_MAX &&
                dist[u] + graph[u][v] < dist[v]) {
                dist[v] = dist[u] + graph[u][v];
            }
        }
    }

    // Print the constructed distance array
    cout << "Charging Station\tDistance from Source\n";
    for (int i = 0; i < V; i++) {
        cout << i << "\t\t" << dist[i] << "\n";
    }
}


vector<int> dijkstra_Algorithm(int V, vector<vector<pair<int, int>>>& adjList, int source) {
    vector<int> dist(V, INT_MAX);  
    priority_queue<pair<int, int>, vector<pair<int, int>>, greater<pair<int, int>>> pq;

    dist[source] = 0; 
    pq.push({0, source});

    while (!pq.empty()) {
        int u = pq.top().second;
        pq.pop();

        for (const auto& edge : adjList[u]) {
            int v = edge.first;    // Destination vertex
            int weight = edge.second; // Edge weight (distance)

            if (dist[u] + weight < dist[v]) {
                dist[v] = dist[u] + weight;
                pq.push({dist[v], v});
            }
        }
    }

    return dist;
}


// Function to implement Selection Sort in descending order
void selectionSortDescending(vector<pair<int, int>>& employees) {
    int n = employees.size();
    
    // Traverse through all elements in the vector
    for (int i = 0; i < n - 1; i++) {
        int maxIndex = i;
        
        // Find the maximum element in unsorted part of array
        for (int j = i + 1; j < n; j++) {
            if (employees[j].second > employees[maxIndex].second) {
                maxIndex = j;
            }
        }
        
        // Swap the found maximum element with the first element
        swap(employees[i], employees[maxIndex]);
    }
}


int partition(vector<pair<int, string>>& properties, int low, int high) {
    int pivot = properties[high].first; 
    int i = low - 1; 

    // Reorder the elements based on the pivot
    for (int j = low; j < high; j++) {
        if (properties[j].first <= pivot) {  // If the current price is smaller or equal to pivot
            i++;
            swap(properties[i], properties[j]);
        }
    }
    swap(properties[i + 1], properties[high]);
    return i + 1; 
}

void quickSort(vector<pair<int, string>>& properties, int low, int high) {
    if (low < high) {
        int pivotIndex = partition(properties, low, high); 
        quickSort(properties, low, pivotIndex - 1); 
        quickSort(properties, pivotIndex + 1, high);   
    }
}

void print(const vector<pair<int, string>>& properties) {
    for (const auto& property : properties) {
        cout << "Price: " << property.first << ", Location: " << property.second << endl;
    }
}


void merge(vector<pair<int, string>>& properties, int left, int mid, int right) {
    int n1 = mid - left + 1;
    int n2 = right - mid;

    // Create temporary vectors
    vector<pair<int, string>> leftVec(n1), rightVec(n2);

    // Copy data
    for (int i = 0; i < n1; i++) {
        leftVec[i] = properties[left + i];
    }
    for (int i = 0; i < n2; i++) {
        rightVec[i] = properties[mid + 1 + i];
    }

  // (Ascending Order)
    int i = 0, j = 0, k = left;
    while (i < n1 && j < n2) {
        if (leftVec[i].first <= rightVec[j].first) {
            properties[k++] = leftVec[i++];
        } else {
            properties[k++] = rightVec[j++];
        }
    }

    // Copy the remaining elements of leftVec
    while (i < n1) {
        properties[k++] = leftVec[i++];
    }

    // Copy the remaining elements of rightVec
    while (j < n2) {
        properties[k++] = rightVec[j++];
    }
}


void mergeSort(vector<pair<int, string>>& properties, int left, int right) {
    if (left < right) {
        int mid = left + (right - left) / 2;

        mergeSort(properties, left, mid);
        mergeSort(properties, mid + 1, right);
        merge(properties, left, mid, right);
    }
}


void printProperties(const vector<pair<int, string>>& properties) {
    for (const auto& property : properties) {
        cout << "Price: " << property.first << ", Location: " << property.second << endl;
    }
}


// Function to compute the LPS array for the pattern
void computeLPSArray(const string& pattern, vector<int>& lps) {
    int length = 0; // length of the previous longest prefix suffix
    int i = 1;
    lps[0] = 0; // The LPS array for the first character is always 0

    while (i < pattern.length()) {
        if (pattern[i] == pattern[length]) {
            length++;
            lps[i] = length;
            i++;
        } else {
            if (length != 0) {
                length = lps[length - 1]; // Try the previous longest prefix suffix
            } else {
                lps[i] = 0;
                i++;
            }
        }
    }
}

// Function to perform KMP search
void KMPSearch(const string& text, const string& pattern) {
    int M = pattern.length();
    int N = text.length();

    // Create the LPS array for the pattern
    vector<int> lps(M);
    computeLPSArray(pattern, lps);

    int i = 0; // index for text
    int j = 0; // index for pattern

    while (i < N) {
        if (pattern[j] == text[i]) {
            i++;
            j++;
        }

        if (j == M) {
            cout << "Pattern found at index " << i - j << endl;
            j = lps[j - 1];
        } else if (i < N && pattern[j] != text[i]) {
            if (j != 0) {
                j = lps[j - 1];
            } else {
                i++;
            }
        }
    }
}


// Floyd-Warshall Algorithm to find shortest paths between all pairs
void floydWarshall(vector<vector<int>>& graph) {
    int V = graph.size();
    vector<vector<long long>> dist(V, vector<long long>(V, numeric_limits<long long>::max()));

    // Initialize the distance matrix with graph values
    for (int i = 0; i < V; i++) {
        for (int j = 0; j < V; j++) {
            if (i == j) 
                dist[i][j] = 0; // Distance to itself is zero
            else if (graph[i][j] != numeric_limits<int>::max()) 
                dist[i][j] = graph[i][j];
        }
    }

    // Apply Floyd-Warshall Algorithm
    for (int k = 0; k < V; k++) {
        for (int i = 0; i < V; i++) {
            for (int j = 0; j < V; j++) {
                if (dist[i][k] != numeric_limits<long long>::max() && 
                    dist[k][j] != numeric_limits<long long>::max() &&
                    dist[i][j] > dist[i][k] + dist[k][j]) {
                    dist[i][j] = dist[i][k] + dist[k][j];
                }
            }
        }
    }

    // Print the shortest path distance matrix
    cout << "Shortest Path Distance Matrix:" << endl;
    for (int i = 0; i < V; i++) {
        for (int j = 0; j < V; j++) {
            if (dist[i][j] == numeric_limits<long long>::max())
                cout << "INF ";
            else
                cout << dist[i][j] << " ";
        }
        cout << endl;
    }
}


// Function to perform Selection Sort
void selectionSort(int arr[], int n) {
    for (int i = 0; i < n - 1; i++) {
        // Find the minimum element in unsorted array
        int minIdx = i;
        for (int j = i + 1; j < n; j++) {
            if (arr[j] < arr[minIdx]) {
                minIdx = j;
            }
        }
        // Swap the found minimum element with the first element
        swap(arr[minIdx], arr[i]);
    }
}



struct Edge {
    int source;       // starting node of the edge
    int destination;  // ending node of the edge
    int weight;   // weight of the edge
};

void dijkstra(const unordered_map<int, vector<Edge>>& graph, int startNode, int endNode, const unordered_map<int, string>& locations) {
    // Step 1: Set up distances and predecessors
    unordered_map<int, int> dist;
    unordered_map<int, int> parent;
    for (const auto& node : graph) {
        dist[node.first] = INT_MAX;  // Initialize distances to infinity
    }
    dist[startNode] = 0;  // Distance to start node is 0
    
    set<pair<int, int>> pq;  // Min-heap to store (distance, node) pairs
    pq.insert({0, startNode});
    
    while (!pq.empty()) {
        // Step 2: Extract the node with the smallest distance
        int u = pq.begin()->second;
        pq.erase(pq.begin());
        
        // Step 3: Explore neighbors of node u
        for (const auto& edge : graph.at(u)) {
            int v = edge.destination;
            int weight = edge.weight;
            
            if (dist[u] + weight < dist[v]) {
                // Update the distance and add the neighbor to the priority queue
                pq.erase({dist[v], v});  // Remove old entry
                dist[v] = dist[u] + weight;
                parent[v] = u;
                pq.insert({dist[v], v});
            }
        }
    }
    
    // Step 4: Output the result
    if (dist[endNode] == INT_MAX) {
        cout << "No path found from " << locations.at(startNode) << " to " << locations.at(endNode) << ".\n";
        return;
    }
    
    cout << "Shortest path from " << locations.at(startNode) << " to " << locations.at(endNode) << ":\n";
    cout << "Distance: " << dist[endNode] << "\n";
    
    // Reconstruct the path
    vector<int> path;
    for (int v = endNode; v != startNode; v = parent[v]) {
        path.push_back(v);
    }
    path.push_back(startNode);
    
    // Print the path
    cout << "Path: ";
    for (int i = path.size() - 1; i >= 0; i--) {
        cout << locations.at(path[i]);
        if (i != 0) cout << " -> ";
    }
    cout << "\n";
}

void unionSets(vector<int> &arr, int n, int u, int v) {
    int temp = arr[u];
    for (int i = 0; i < n; i++) {
        if (arr[i] == temp) {
            arr[i] = arr[v];
        }
    }
}

int findSet(vector<int> &arr, int u, int v) {
    return arr[u] != arr[v];  // Corrected condition
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
        int u = edges[i].source;       // Using 'source' as the starting node of the edge
        int v = edges[i].destination;  // Using 'destination' as the ending node of the edge
        
        if (findSet(arr, u, v)) {  // Corrected condition to check if u and v are in different sets
            mst.push_back(edges[i]);  // Add the edge to the MST
            unionSets(arr, n, u, v);   // Assuming 'unionSets' unites the sets of u and v
        }
    }
    
    return mst;
}

void bellmanFord(int V, int E, vector<Edge>& edges, int source) {
    vector<int> distance(V, INT_MAX);
    distance[source] = 0;

    for (int i = 0; i < V - 1; ++i) {
        for (int j = 0; j < E; ++j) {
            int u = edges[j].source;
            int v = edges[j].destination;
            int weight = edges[j].weight;

            if (distance[u] != INT_MAX && distance[u] + weight < distance[v]) {
                distance[v] = distance[u] + weight;
            }
        }
    }

    for (int j = 0; j < E; ++j) {
        int u = edges[j].source;
        int v = edges[j].destination;
        int weight = edges[j].weight;

        if (distance[u] != INT_MAX && distance[u] + weight < distance[v]) {
            cout << "Graph contains a negative-weight cycle.\n";
            return;
        }
    }

    cout << "Vertex\tDistance from Source\n";
    for (int i = 0; i < V; ++i) {
        cout << i << "\t" << distance[i] << "\n";
    }
}

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

struct College {
    int id;
    int cityLevel;
};

void heapify(vector<College>& colleges, int n, int i) {
    int largest = i;  // Initialize the largest as the root
    int left = 2 * i + 1;  // Left child
    int right = 2 * i + 2;  // Right child

    if (left < n && (colleges[left].cityLevel > colleges[largest].cityLevel ||
                     (colleges[left].cityLevel == colleges[largest].cityLevel && colleges[left].id > colleges[largest].id))) {
        largest = left;
    }

    if (right < n && (colleges[right].cityLevel > colleges[largest].cityLevel ||
                      (colleges[right].cityLevel == colleges[largest].cityLevel && colleges[right].id > colleges[largest].id))) {
        largest = right;
    }

    if (largest != i) {
        swap(colleges[i], colleges[largest]);
        heapify(colleges, n, largest);
    }
}

void heapSort(std::vector<College>& colleges) {
    int n = colleges.size();

    for (int i = n / 2 - 1; i >= 0; i--) {
        heapify(colleges, n, i);
    }

    for (int i = n - 1; i > 0; i--) {
        swap(colleges[0], colleges[i]);
        heapify(colleges, i, 0);
    }
}

struct University {
    int universityId;
    string universityName;
    int cityTier;
};

void heapify(vector<University>& universities, int size, int idx) {
    int largest = idx;  
    int leftChild = 2 * idx + 1;  
    int rightChild = 2 * idx + 2;  

    if (leftChild < size && (universities[leftChild].cityTier > universities[largest].cityTier ||
                             (universities[leftChild].cityTier == universities[largest].cityTier && universities[leftChild].universityId > universities[largest].universityId))) {
        largest = leftChild;
    }

    if (rightChild < size && (universities[rightChild].cityTier > universities[largest].cityTier ||
                              (universities[rightChild].cityTier == universities[largest].cityTier && universities[rightChild].universityId > universities[largest].universityId))) {
        largest = rightChild;
    }

    if (largest != idx) {
        swap(universities[idx], universities[largest]);
        heapify(universities, size, largest);
    }
}

void heapSort(vector<University>& universities) {
    int totalUniversities = universities.size();

    for (int i = totalUniversities / 2 - 1; i >= 0; i--) {
        heapify(universities, totalUniversities, i);
    }

    for (int i = totalUniversities - 1; i > 0; i--) {
        swap(universities[0], universities[i]);
        heapify(universities, i, 0);
    }
}

struct Institution {
    int institutionId;
    string institutionName;
    int cityRank;
};

void showInstitutionsByLevel(const vector<Institution>& institutions, int level) {
    queue<Institution> institutionQueue;

    for (const auto& institution : institutions) {
        institutionQueue.push(institution);
    }

    cout << "Institutions in City Rank " << level << ":\n";
    while (!institutionQueue.empty()) {
        Institution currentInstitution = institutionQueue.front();
        institutionQueue.pop();

        if (currentInstitution.cityRank == level) {
            cout << "Institution Name: " << currentInstitution.institutionName << " | Rank: " << currentInstitution.cityRank << endl;
        }
    }
}


int main() {
    cout << "1-Sumit 2-Vinay 3-Shashank 4-Rahul" << endl;

    int choice;
    cin >> choice;

    switch (choice) {
        case 1: {
            // Kruskal's Algorithm
            cout << "Kruskal's Algorithm" << endl;

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
          for (const auto& e : edges) {
    cout << "(" << locations[e.source] << ", " << locations[e.destination] << ") --> " << e.weight << "\n";
}

cout << "\nEdges in the MST:\n";
int totalCost = 0;
for (const auto& e : mst) {
    cout << "(" << locations[e.source] << ", " << locations[e.destination] << ") --> " << e.weight << "\n";
    totalCost += e.weight;
}


            cout << "\nMinimum Cost of the MST: " << totalCost << "\n";

            cout << endl << endl << endl;

            // Bellman-Ford Algorithm
            cout << "Bellman Ford Algorithm" << endl;

            int V = 5;
            int E = 8;

            vector<Edge> bellmanEdges = {
                {0, 1, -1}, {0, 2, 4}, {1, 2, 3},
                {1, 3, 2}, {1, 4, 2}, {3, 2, 5},
                {3, 1, 1}, {4, 3, -3}
            };

            int source = 0;
            bellmanFord(V, E, bellmanEdges, source);

            cout << endl << endl << endl;

            // Rabin-Karp Algorithm
            cout << "Rabin Karp Algorithm" << endl;

            string text = "Batch001QualityOKBatch002SpoiledBatch003QualityOK";
            string pattern = "QualityOK";
            int prime = 101;

            cout << "Text: " << text << endl;
            cout << "Pattern: " << pattern << endl;

            rabin_karp(text, pattern, prime);

            cout << endl << endl << endl;

            // Fenwick Tree
            cout << "Fenwick Tree" << endl;

            vector<int> water_levels = {50, 70, 30, 80, 45, 60, 75};
            Fenwicktree fenwick(water_levels.size());
            for (int i = 0; i < water_levels.size(); i++) {
                fenwick.update(i + 1, water_levels[i]);
            }

            cout << "Total water level in reservoirs 1 to 3: "
                 << fenwick.range_query(1, 3) << " million liters" << endl;

            fenwick.update(3, 10);
            cout << "Updated total water level in reservoirs 1 to 3: "
                 << fenwick.range_query(1, 3) << " million liters" << endl;

            cout << "Total water level in reservoirs 1 to 7: "
                 << fenwick.range_query(1, 7) << " million liters" << endl;

            break;
        }
        case 2: {
            // Prgm 1

            vector<College> colleges = {
                {101, 2}, {102, 1}, {103, 3}, {104, 2}, {105, 1}
            };

            // Apply heapsort
            heapSort(colleges);

            // Display the sorted colleges
            cout << "Sorted colleges (by city level, then ID):\n";
            for (const auto& college : colleges) {
                cout << "College ID: " << college.id << ", City Level: " << college.cityLevel << "\n";
            }

            cout << endl << endl << endl;

            // Prgm 2

            vector<University> universities = {
                {201, "Indian Institute of Technology", 2},
                {202, "kle tu", 1},
                {203, "Delhi University", 3},
                {204, "Jawaharlal Nehru University", 2},
                {205, "National Institute of Technology", 1},
                {206, "Banaras Hindu University", 3},
                {207, "Anna University", 2},
                {208, "kle tech", 3}
            };

            // Apply heapsort
            heapSort(universities);

            // Display the sorted universities
            cout << "Sorted universities (by city tier, then ID):\n";
            for (const auto& university : universities) {
                cout << "University ID: " << university.universityId
                     << ", Name: " << university.universityName
                     << ", City Tier: " << university.cityTier << "\n";
            }

            cout << endl << endl << endl;

            // Prgm 3
            vector<Institution> institutions = {
                {201, "kle tu", 1},
                {202, "Indian Institute of Science", 2},
                {203, "D University", 3},
                {204, "Jawaharlal Nehru University", 1},
                {205, "National Institute of Technology", 2},
                {206, "Banaras Hindu University", 3},
                {207, "Anna University", 1},
                {208, "kle it", 3}
            };

            // Display institutions at city rank 1
            showInstitutionsByLevel(institutions, 1);

            cout << endl << endl << endl;

            // Prgm 4
            unordered_map<int, vector<Edge>> graph = {
    {0, {{0, 1, 4}, {0, 2, 1}}},
    {1, {{1, 0, 4}, {1, 2, 2}, {1, 3, 5}}},
    {2, {{2, 0, 1}, {2, 1, 2}, {2, 3, 8}, {2, 4, 10}}},
    {3, {{3, 1, 5}, {3, 2, 8}, {3, 4, 2}}},
    {4, {{4, 2, 10}, {4, 3, 2}}}
};

            // Define place names
            unordered_map<int, string> locations = {
                {0, "Market"},
                {1, "Hospital"},
                {2, "Library"},
                {3, "School"},
                {4, "Park"}
            };

            // Default source and destination
            int startNode = 0; // "Market" ID
            int endNode = 4; // "Park" ID

            // Run Dijkstra's algorithm
            dijkstra(graph, startNode, endNode, locations);

            break;
        }
        case 3:{
            //Prgm1
            cout<<"Merge Sort"<<endl;
            vector<pair<int, string>> properties = {
        {500000, "LocationA"},
        {300000, "LocationB"},
        {400000, "LocationC"},
        {600000, "LocationD"},
        {200000, "LocationE"}
    };

    //merge sort
    mergeSort(properties, 0, properties.size() - 1);
     cout << "Sorted Properties (Ascending by Price):" << endl;
     printProperties(properties);
   
   
     reverse(properties.begin(), properties.end());
    // descending order by price
    cout <<"\nSorted Properties (Descending by Price):"<< endl;
    printProperties(properties);

            cout<<endl;
            cout<<endl;
            cout<<endl;

            //Prgm 2
            cout<<"Quick sort"<<endl;

                vector<pair<int, string>> high_property = {
        {500000, "LocationA"},
        {300000, "LocationB"},
        {400000, "LocationC"},
        {600000, "LocationD"},
        {200000, "LocationE"}
    };

    quickSort(high_property, 0, high_property.size() - 1);
    cout << "Sorted Properties (Ascending by Price):" << endl;
    print(high_property);

    // Now reverse
    reverse(high_property.begin(), high_property.end());
    cout << "\nSorted Properties (Descending by Price):" << endl;
    printProperties(high_property);
            cout<<endl;
            cout<<endl;
            cout<<endl;

            //Prgm 3
            cout<<"Selection Sort"<<endl;
                // Input: Vector of employee pairs (id, experience)
    vector<pair<int, int>> employees = {
        {101, 5},  // Employee with ID 101 and experience 5 years
        {102, 2},  // Employee with ID 102 and experience 2 years
        {103, 8},  // Employee with ID 103 and experience 8 years
        {104, 3},  // Employee with ID 104 and experience 3 years
        {105, 6}   // Employee with ID 105 and experience 6 years
    };
    
    // Sorting the employees by experience (in descending order) using Selection Sort
    selectionSortDescending(employees);
    
    // Output: Display sorted employees by experience in descending order
    cout << "Employees sorted by experience (descending):" << endl;
    for (const auto& emp : employees) {
        cout << "ID: " << emp.first << ", Experience: " << emp.second << " years" << endl;
    }

            cout<<endl;
            cout<<endl;
            cout<<endl;

            //Prgm 4
            cout<<"Dijkstra Algorithm"<<endl;

             vector<string> places = {
        "Museum", "Park", "Resort", "Hill Trek", "Jungle Safari"
    };

    int V = 5;

    vector<vector<pair<int, int>>> adjList(V);

    adjList[0].push_back({1, 10});  // M- P
    adjList[1].push_back({0, 10});  // P - M
    adjList[0].push_back({2, 20});  // M- R
    adjList[2].push_back({0, 20});  // R- M
    adjList[2].push_back({3, 30});  // R- H
    adjList[3].push_back({2, 30});  // H - R
    adjList[3].push_back({4, 40});  // H- J
    adjList[4].push_back({3, 40});  // J- H
    adjList[4].push_back({1, 15});  // J- P
    adjList[1].push_back({4, 15});  // P- J
    adjList[4].push_back({2, 25});  // J - R
    adjList[2].push_back({4, 25});  // R- J

    cout << "Enter the starting point (0: Museum, 1: Park, 2: Resort, 3: Hill Trek, 4: Jungle Safari): ";
    int start;
    cin >> start;

    vector<int> distances = dijkstra_Algorithm(V, adjList, start);


    cout << "Shortest path from " << places[start] << ":\n";
    for (int i = 0; i < V; i++) {
            cout << "Distance to " << places[i] << ": " << distances[i] << " km\n";
    }

            break;
        }
        case 4:{
            //Prgm 1
            cout<<"Selection Sort"<<endl;
                int chargingStations[] = {50, 30, 20, 40, 60}; // Example distances (in kilometers)
    int n = sizeof(chargingStations) / sizeof(chargingStations[0]);

    cout << "Charging Stations before sorting: ";
    for (int i = 0; i < n; i++) {
        cout << chargingStations[i] << " ";
    }

    selectionSort(chargingStations, n);

    cout << "\nCharging Stations after sorting: ";
    for (int i = 0; i < n; i++) {
        cout << chargingStations[i] << " ";
    }

    cout<<endl;
    cout<<endl;
    cout<<endl;

    //Prgm 2
            cout<<"Floyd warshall Algorithm"<<endl;

      vector<vector<int>> graph = {
        {0, 3, numeric_limits<int>::max(), 7},
        {3, 0, 2, numeric_limits<int>::max()},
        {numeric_limits<int>::max(), 2, 0, 4},
        {7, numeric_limits<int>::max(), 4, 0}
    };

    floydWarshall(graph);

    cout<<endl;
    cout<<endl;
    cout<<endl;
   
        //Prgm 3
        cout<<"KMP Algorithm"<<endl;
         
            // Example: Finding a vehicle license plate in a parking lot record
    string parkingLot = "ABC123, XYZ456, ABC123, DEF789, ABC123";
    string licensePlate = "ABC123";

    // Perform KMP search
    KMPSearch(parkingLot, licensePlate);
   
    cout<<endl;
    cout<<endl;
    cout<<endl;

        //Prgm 4
    cout<<"Dijkstra Algorithm"<<endl;

    vector<vector<int>> graph_mark = {
        {0, 10, 0, 0, 0, 20},
        {10, 0, 30, 0, 0, 0},
        {0, 30, 0, 50, 0, 0},
        {0, 0, 50, 0, 60, 0},
        {0, 0, 0, 60, 0, 10},
        {20, 0, 0, 0, 10, 0}
    };

    int sourceStation = 0;  // Start from charging station 0 (can be set dynamically)

    shortestPathDijkstra(graph_mark, sourceStation);  // Run Dijkstra's Algorithm

            break;
        }
    }
 
    return 0;
}