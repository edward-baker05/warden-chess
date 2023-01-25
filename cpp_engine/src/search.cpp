// Implementation of the search algorithms

#include <algorithm>
#include <limits>
#include <vector>

#include "board.h"
#include "eval.h"
#include "movegen.h"

const int SIMULATIONS_PER_MOVE = 1000;

class Node {
public:
    Node(std::shared_ptr<Board> board, Move move);
    void expand();
    bool isLeaf() const;
    bool isFullyExpanded() const;
    int evaluate() const;
    std::shared_ptr<Node> getBestChild();
    Move getMove() const;
    int getValue() const;
    void setValue(int value);

private:
    std::shared_ptr<Board> board;
    Move move;
    int value;
    std::vector<std::shared_ptr<Node>> children;
};

void MCTS(std::shared_ptr<Node> root, int depth, int alpha, int beta) {
    if (depth == 0 || root->isLeaf()) {
        // Leaf node, return the heuristic evaluation
        return root->evaluate();
    }

    // Expand the children of the current node if it hasn't been fully expanded yet
    if (!root->isFullyExpanded()) {
        root->expand();
    }

    int value;
    if (root->isMaxNode()) {
        value = std::numeric_limits<int>::min();
        for (const auto& child : root->children) {
            int childValue = MCTS(child, depth - 1, alpha, beta);
            value = std::max(value, childValue);
            alpha = std::max(alpha, value);
            if (beta <= alpha) {
                break;
            }
        }
    } else {
        value = std::numeric_limits<int>::max();
        for (const auto& child : root->children) {
            int childValue = MCTS(child, depth - 1, alpha, beta);
            value = std::min(value, childValue);
            beta = std::min(beta, value);
            if (beta <= alpha) {
                break;
            }
        }
    }

    return value;
}

Move search(std::shared_ptr<Board> board) {
    // Initialize the root node with the current state of the board
    auto root = std::make_shared<Node>(board, Move());

    // Run MCTS with alpha-beta pruning for a certain number of simulations
    const int SIMULATIONS_PER_MOVE = 1000;
    for (int i = 0; i < SIMULATIONS_PER_MOVE; i++) {
        int value = MCTS(root, MAX_DEPTH, std::numeric_limits<int>::min(), std::numeric_limits<int>::max());
    }

    // Select the child node with the highest value
    auto bestChild = root->getBestChild();

    // Return the move that leads to the best child node
    return bestChild->getMove();
}

Node::Node(std::shared_ptr<Board> board, Move move) : board(board), move(move), value(0) {
    // Initialize the node with the current state of the board
}

void Node::expand() {
    // Expand the children of the current node
    std::vector<Move> moves = Movegen::generateMoves(board);
    for (const auto& move : moves) {
        auto newBoard = std::make_shared<Board>(*board);
        newBoard->makeMove(move);
        children.emplace_back(std::make_shared<Node>(newBoard, move));
    }
}

bool Node::isLeaf() const {
    // Check if the current node is a leaf node
    return children.empty();
}

bool Node::isFullyExpanded() const {
    // Check if the current node has been fully expanded
    return children.size() == Movegen::generateMoves(board).size();
}

int Node::evaluate() const {
    // Return the heuristic evaluation of the current node
    return Eval::evaluate(board);
}

std::shared_ptr<Node> Node::getBestChild() {
    int maxValue = std::numeric_limits<int>::min();
    std::shared_ptr<Node> bestChild = nullptr;
    for (const auto& child : children) {
        if (child->getValue() > maxValue) {
            maxValue = child->getValue();
            bestChild = child;
        }
    }
    return bestChild;
}

Move Node::getMove() const {
    return move;
}

int Node::getValue() const {
    return value;
}
