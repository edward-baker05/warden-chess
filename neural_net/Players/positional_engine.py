import chess
from random import choices
from typing import List

class PositionalEngine:
    def __init__(self, colour: bool = True, depth: int = 3):
        self.piece_values = {'1': 100.0, '2': 300.0, '3': 350.0, '4': 500.0, '5': 900.0, '6': 10000.0}
        self.colour = colour
        self.depth = depth  # depth of minimax search
        self.time_limit = 20  # time limit for minimax search

    def evaluate(self, board: chess.Board) -> float:
        evaluation = self.count_material(board)
        perspective = 1 if self.colour else -1

        return evaluation * perspective

    def count_material(self, board: chess.Board) -> float:
        """Count the material on the board"""
        material = 0.0
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece is not None:
                if piece.color == self.colour:
                    material += self.piece_values[str(piece.piece_type).lower()]
                else:
                    material -= self.piece_values[str(piece.piece_type).lower()]
        return material

    def search(self, board: chess.Board, depth: int, alpha: float, beta: float) -> float:
        if depth == 0:
            return self.search_captures(board, 0, alpha, beta)

        moves = list(board.legal_moves)
        if len(moves) == 0:
            if board.is_checkmate():
                return float("-inf")
            return 0.0

        for move in self.order_moves(moves, board):
            temp_board = board.copy()
            temp_board.push(move)
            evaluation = -self.search(temp_board, depth - 1, -beta, -alpha)
            if evaluation >= beta:
                return beta
            alpha = max(alpha, evaluation)

        return alpha

    def order_moves(self, moves: List[chess.Move], board: chess.Board):
        if len(moves) == 0:
            return []

        ordered = {}
        for move in moves:
            move_score_guess = 0
            move_piece_type = str(board.piece_at(move.from_square).piece_type).lower()
            try:
                capture_piece_type = str(board.piece_at(move.to_square).piece_type).lower()
                
                # Prioritise capturing pieces with invaluable pieces
                if capture_piece_type != "none":
                    move_score_guess = 10 * self.piece_values[capture_piece_type] - self.piece_values[move_piece_type]
            except AttributeError:
                pass

            # Check if it is promoting a pawn, as this is likely a good idea
            if move.promotion is not None:
                move_score_guess += self.piece_values[str(move.promotion).lower()]

            ordered[move] = move_score_guess

        ordered_moves = list(dict(sorted(ordered.items(), key=lambda item: item[1], reverse=True)).keys())
        return ordered_moves

    def search_captures(self, board: chess.Board, depth: int, alpha: float, beta: float) -> float:
        evaluation = self.evaluate(board)
        if evaluation >= beta:
            return beta
        alpha = max(alpha, evaluation)

        capture_moves = [move for move in board.legal_moves if board.is_capture(move)]
        capture_moves = self.order_moves(capture_moves, board)

        for move in capture_moves:
            temp_board = board.copy()
            temp_board.push(move)
            evaluation = -self.search_captures(temp_board, depth + 1, -beta, -alpha)

            if evaluation >= beta:
                return beta
            alpha = max(alpha, evaluation)

        return alpha

    def get_move(self, board: chess.Board) -> chess.Move:
        moves = list(board.legal_moves)
        best_moves = []
        best_evaluation = float("-inf")

        for move in moves:
            temp_board = board.copy()
            temp_board.push(move)

            # Check if its checkmate 
            if temp_board.is_checkmate():
                return move

            evaluation = -self.search(temp_board, self.depth - 1, float("-inf"), float("inf"))

            if evaluation > best_evaluation:
                best_evaluation = evaluation
                best_moves = [move]
                print("New best move found:", move.uci(), "with evaluation", best_evaluation)
            elif evaluation == best_evaluation:
                best_moves.append(move)

        from numpy import linspace
        
        weights = list(linspace(1, 0, len(best_moves)))
        best_move = choices(best_moves, weights=weights)[0]

        print(f"AI move made: {best_move.uci()} with advantage {best_evaluation}, filtered from {len(list(moves))} moves")
        return best_move
