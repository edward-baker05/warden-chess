// Header file for evaluation functions

#ifndef EVAL_H
#define EVAL_H

#include "board.h"

class Eval {
public:
    static int materialScore(Board* board);
    static int pieceSquareTable(Board* board);
    static int mobilityScore(Board* board);
    static int kingSafetyScore(Board* board);
    static int evaluate(Board* board);

private:
    static const int PawnTable[64];
    static const int KnightTable[64];
    static const int BishopTable[64];
    static const int RookTable[64];
    static const int QueenTable[64];
    static const int KingTable[64];
    static const int KnightMobility[32];
    static const int BishopMobility[32];
    static const int RookMobility[32];
    static const int QueenMobility[32];
    static const int KingSafety[9];
};

#endif
