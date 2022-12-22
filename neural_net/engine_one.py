# Engine which plays the move with the highest immediate material advantage
class MaterialEngine():
    def __init__(self, input_color):

      # Creates piece value scheme to specify value of each piece.
      # Pawn = 1, Knight = 3, Bishop = 3, Rook = 5, Queen = 9, King = 99
      self.piece_values = (1, 3, 3, 5, 9, 99)

      # Super call to
      super().__init__(input_color)

    def generate_move(self, position):
      # Position parameter is an object of type Board

      # Finds all possible moves it can play.
      moves = position.all_possible_moves(self.color)

      # Initalizes best move and advantage after it has been played to dummy values.
      best_move = None
      best_move_advantage = -99

      # Loops through possible moves
      for move in moves:
        """ advantage_as_result(move, piece_values) finds numerical advantage
        as specified by piece value scheme above. Returns negative values for
        positions of disadvantage. Returns +/-99 for checkmate. """
        advantage = position.advantage_as_result(move, self.piece_values)

        # If this move is better than best move, it is the best move.
        if advantage >= best_move_advantage:
            best_move = move
            best_move_advantage = advantage

      return best_move