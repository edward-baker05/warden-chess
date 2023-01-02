import chess
from random import choice

"""Engine which plays the move with the highest immediate material advantage"""
class MaterialEngine:
    def __init__(self, colour: bool, depth: int = 4):
        """Creates piece value scheme to specify value of each piece.
        Pawn = 1, Knight = 3, Bishop = 3, Rook = 5, Queen = 9, King = inf"""
        self.piece_values = [1.0, 3.0, 3.5, 5.0, 9.0, float("inf")]
        self.colour = colour
        self.depth = depth  # depth of minimax search
        # self.time_limit = 20  # time limit for minimax search

    def get_move(self, board: chess.Board) -> tuple[chess.Move, float]:
        # Position parameter is an object of type Board

        # Finds all possible moves it can play.
        moves = board.legal_moves

        # Initalizes best move and advantage after it has been played to dummy values.
        choices = []
        best_move_advantage = float('inf')

        # Initialize alpha and beta to negative and positive infinity, respectively
        alpha = -float('inf')
        beta = float('inf')

        # Loops through possible moves
        for move in moves:
            advantage = -self.minimax(board, move, self.depth, alpha, beta, True)
            if board.gives_check(move):
                advantage +=  0.5

            # If this move is better than best move, it is the best move.
            if advantage > best_move_advantage:
                choices = [move]
                best_move_advantage = advantage
                alpha = max(alpha, best_move_advantage)
            elif advantage == best_move_advantage:
                choices.append(move)

        move = choice(choices)
        print(f"AI move made: {move.uci()} with advantage {best_move_advantage} out of {len(choices)} choices, filtered from {len(list(moves))} moves")
        return move # type: ignore

    def minimax(self, board: chess.Board, move: chess.Move, depth: int, alpha: float, beta: float, is_max_player: bool) -> float:
        """
        Recursive function that implements the minimax algorithm with alpha-beta pruning
        """
        test_board = board.copy()
        test_board.push(move)

        # base case: reached the depth limit or the game is over
        if depth == 0 or test_board.is_game_over():
            return self.material_advantage(test_board)

        if is_max_player:
            best_value = -float('inf')
            for next_move in test_board.legal_moves:
                value = self.minimax(test_board, next_move, depth - 1, alpha, beta, False)
                best_value = max(best_value, value)
                alpha = max(alpha, best_value)
                if beta <= alpha:
                    break  # prune the search
            return best_value
        else:
            best_value = float('inf')
            for next_move in test_board.legal_moves:
                value = self.minimax(test_board, next_move, depth - 1, alpha, beta, True)
                best_value = min(best_value, value)
                beta = min(beta, best_value)
                if beta <= alpha:
                    break  # prune the search
            return best_value

    def material_advantage(self, board: chess.Board) -> float:
        """
        Finds the advantage a particular side possesses given a value scheme.
        """
        if board.is_game_over():
            outcome = board.outcome()
            termination = outcome.termination
            if termination.value == 1:
                return float("inf") if outcome.winner == self.colour else float("-inf")
            else:
                return 0.0

        advantage = sum([len(board.pieces(piece, self.colour)) * self.piece_values[piece-1] for piece in range(1, 7)]) - sum([len(board.pieces(piece, not self.colour)) * self.piece_values[piece-1] for piece in range(1, 7)])

        return advantage
