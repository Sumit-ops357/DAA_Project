#include<iostream>
#include <cstdlib>
#include <string>
using namespace std;

typedef struct farm {
    string location;
    string cropTypes;
    int landArea;
} FARM;

typedef struct tree {
    FARM farmData;
    struct tree *left;
    struct tree *right;
} TREE;

class binarysearchtree {
    public:
    TREE *insert_to_bst(TREE*, FARM);
    void inorder_traversal(TREE*);
    void preorder_traversal(TREE*);
    void postorder_traversal(TREE*);
    TREE *delete_from_bst(TREE*, int);
};

TREE *binarysearchtree::insert_to_bst(TREE *root, FARM farmData) {
    TREE *newnode = new TREE();
    newnode->farmData = farmData;
    newnode->left = newnode->right = NULL;

    if (root == NULL) {
        return newnode;
    }

    TREE *curr = root;
    TREE *parent = NULL;

    while (curr != NULL) {
        parent = curr;
        if (farmData.landArea < curr->farmData.landArea)
            curr = curr->left;
        else
            curr = curr->right;
    }

    if (farmData.landArea < parent->farmData.landArea)
        parent->left = newnode;
    else
        parent->right = newnode;

    return root;
}

void binarysearchtree::inorder_traversal(TREE *root) {
    if (root != NULL) {
        inorder_traversal(root->left);
        cout << "Location: " << root->farmData.location << ", Crops: " << root->farmData.cropTypes << ", Land Area: " << root->farmData.landArea << endl;
        inorder_traversal(root->right);
    }
}

void binarysearchtree::preorder_traversal(TREE *root) {
    if (root != NULL) {
        cout << "Location: " << root->farmData.location << ", Crops: " << root->farmData.cropTypes << ", Land Area: " << root->farmData.landArea << endl;
        preorder_traversal(root->left);
        preorder_traversal(root->right);
    }
}

void binarysearchtree::postorder_traversal(TREE *root) {
    if (root != NULL) {
        postorder_traversal(root->left);
        postorder_traversal(root->right);
        cout << "Location: " << root->farmData.location << ", Crops: " << root->farmData.cropTypes << ", Land Area: " << root->farmData.landArea << endl;
    }
}

TREE *binarysearchtree::delete_from_bst(TREE *root, int landArea) {
    if (root == NULL) return root;

    if (landArea < root->farmData.landArea)
        root->left = delete_from_bst(root->left, landArea);
    else if (landArea > root->farmData.landArea)
        root->right = delete_from_bst(root->right, landArea);
    else {
        if (root->left == NULL) {
            TREE *temp = root->right;
            delete root;
            return temp;
        }
        else if (root->right == NULL) {
            TREE *temp = root->left;
            delete root;
            return temp;
        }

        TREE *succParent = root;
        TREE *succ = root->right;
        while (succ->left != NULL) {
            succParent = succ;
            succ = succ->left;
        }

        if (succParent != root)
            succParent->left = succ->right;
        else
            succParent->right = succ->right;

        root->farmData = succ->farmData;
        delete succ;
    }

    return root;
}

int main() {
    binarysearchtree tree;
    TREE *root;
    root = NULL;

    FARM farms[] = {
        {"Rooftop Garden", "Tomatoes, Lettuce", 150},
        {"Community Park", "Carrots, Spinach", 300},
        {"Vertical Farm", "Peppers, Cucumbers", 200},
        {"Hydroponic Facility", "Broccoli, Eggplants", 250},
        {"Urban Greenhouse", "Tomatoes, Spinach", 180},
        {"Backyard Farm", "Lettuce, Carrots", 100}
    };

    for (int i = 0; i < 6; i++) {
        root = tree.insert_to_bst(root, farms[i]);
    }

    int choice = 0;
    FARM farmData;

    while (1) {
        cout << "\n*******************************************************************\n";
        cout << "*                         MENU                                    *\n";
        cout << "*******************************************************************\n";
        cout << "1--Insert a node in binary tree                                   *\n";
        cout << "2--Inorder traversal                                              *\n";
        cout << "3--Preorder traversal                                             *\n";
        cout << "4--Postorder traversal                                            *\n";
        cout << "5--Delete a node                                                  *\n";
        cout << "0--EXIT                                                           *\n";
        cout << "*******************************************************************\n";

        cout << "enter your choice\n";
        cin >> choice;

        if (choice == 1) {
            cout << "Enter the farm location: ";
            cin.ignore();
            getline(cin, farmData.location);
            cout << "Enter the crop types: ";
            getline(cin, farmData.cropTypes);
            cout << "Enter the land area: ";
            cin >> farmData.landArea;
            root = tree.insert_to_bst(root, farmData);
        }
        else if (choice == 2) {
            if (root == NULL) {
                cout << "tree is empty\n";
            }
            else {
                cout << "Inorder traversal is...\n";
                tree.inorder_traversal(root);
                cout << endl;
            }
        }
        else if (choice == 3) {
            if (root == NULL) {
                cout << "tree is empty\n";
            }
            else {
                cout << "Preorder traversal is...\n";
                tree.preorder_traversal(root);
                cout << endl;
            }
        }
        else if (choice == 4) {
            if (root == NULL) {
                cout << "tree is empty\n";
            }
            else {
                cout << "Postorder traversal is...\n";
                tree.postorder_traversal(root);
                cout << endl;
            }
        }
        else if (choice == 5) {
            cout << "Enter the land area of the node to be deleted\n";
            cin >> farmData.landArea;
            root = tree.delete_from_bst(root, farmData.landArea);
            cout << endl;
        }
        else if (choice == 0) {
            exit(0);
        }
        else {
            cout << "You have entered the wrong choice\n";
        }
    }
    return 0;
}