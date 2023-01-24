// Header file for board representation and manipulation

#pragma once

#include <vector>
#include "bitboard.h"

enum {WHITE, BLACK, NO_PIECE};

struct Move {
    int from;
    int to;
    int capturedPiece;
    int promotedPiece;
};

class Board {
    public:
        Board();
        ~Board();
        void print() const;
        bool isCheck() const;
        bool isMate() const;
        bool isStaleMate() const;
        void makeMove(Move move);
        void unmakeMove();
        int getPiece(int square) const;
        Bitboard* bb;
        int toMove;
        int fiftyMoveCounter;
        int fullMoveCounter;
        std::vector<Move> moveList;
};
