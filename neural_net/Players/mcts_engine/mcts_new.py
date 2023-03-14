import math

class Node:
    def __init__(self, state, move=None):
        self.state = state
        self.move = move
        self.visit_count = 0
        self.total_score = 0
        self.children = []

def select(node):
    log_n = math.log(node.visit_count)

    def ucb_score(child):
        exploitation = child.total_score / child.visit_count
        exploration = math.sqrt(log_n / child.visit_count)
        return exploitation + C * exploration

    return max(node.children, key=ucb_score)

def expand(node, policy_network):
    # TODO: add a child node for each legal move from the current state,
    # using the policy network to get the probability of each move

def simulate(state):
    # TODO: play a random game from the given state until the end of the game,
    # and return the final score

def backpropagate(node, score):
    # TODO: update the visit count and total score of all nodes in the path
    # from the given node up to the root node

def mcts(root_node, num_simulations, policy_network):
    for i in range(num_simulations):
        node = root_node
        while node.children:
            node = select(node)
        if node.visit_count == 0:
            expand(node, policy_network)
        score = simulate(node.state)
        backpropagate(node)

    # Get the visit counts for each child node and normalize them
    visit_counts = [child.visit_count for child in root_node.children]
    sum_counts = sum(visit_counts)
    if sum_counts == 0:
        return [1 / len(root_node.children)] * len(root_node.children)
    else:
        return [count / sum_counts for count in visit_counts]
