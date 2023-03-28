from __future__ import annotations

import math
from typing import Optional

import chess


class Node:
    def __init__(self, board: chess.Board, parent: Optional[Node] = None) -> None:
        """
        Initialize a node in the Monte Carlo tree.

        Args:
            board: a chess.Board object representing the current board position.
            parent: the parent Node object. If not provided, defaults to None.
        """
        self.board = board
        self.parent = parent
        self.children = []
        self.visits = 0.0
        self.wins = 0.0
        self.player = board.turn
        self.value = 0.0
        
        if parent is not None:
            self.depth = parent.depth + 1
        else:
            self.depth = 0

    def add_child(self, move: chess.Move) -> Node:
        """
        Add a child node to the current node by making a move on the board.

        Args:
            move: a chess.Move object representing the move to make on the board.

        Returns:
            The newly created child Node object.
        """
        new_board = self.board.copy()
        new_board.push(move)
        child = Node(new_board, parent=self)
        self.children.append(child)
        return child

    def ucb1(self, exploration_param=0.2) -> float:
        """
        Calculate and return the UCB1 score for the current node.

        Args:
            exploration_param: a float parameter to control the balance between exploration and exploitation.

        Returns:
            The UCB1 score for the current node.
        """
        # If the node has not been visited, return infinity
        if self.visits == 0:
            return float('inf')
        # 
        return self.value + exploration_param * math.sqrt(math.log(self.parent.visits) / self.visits)
