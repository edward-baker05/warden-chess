// Header file for move generation

#ifndef MOVEGEN_H
#define MOVEGEN_H

#include <vector>
#include "board.h"

class Movegen {
public:
    static void generateMoves(Board* board);
    static void generateCaptures(Board* board);
    static void generateNonCaptures(Board* board);
    static unsigned long long getAttacks(Board* board, int square);
    static unsigned long long getPawnAttacks(unsigned long long pawns, int square);
    static unsigned long long getRookAttacks(unsigned long long rooks, unsigned long long occupied, int square);
    static unsigned long long getBishopAttacks(unsigned long long bishops, unsigned long long occupied, int square);
    static int getPiece(unsigned long long pieces, int square);
};

#endif
