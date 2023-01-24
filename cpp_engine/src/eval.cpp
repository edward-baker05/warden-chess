// Implementation of evaluation function

#include "eval.h"

const int PieceValues[PIECE_TYPE_NB] = { 0, PawnValue, KnightValue, BishopValue, RookValue, QueenValue, KingValue };
const int PawnAdvance[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };
const int KnightMobility[9] = { -30, -20, -10, 0, 10, 20, 30, 40, 50 };
const int BishopMobility[14] = { -30, -20, -10, 0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100 };
const int RookMobility[15] = { -50, -40, -30, -20, -10, 0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100 };
const int QueenMobility[24] = { -90, -80, -70, -60, -50, -40, -30, -20, -10, 0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140 };
const int KingSafety[12] = { -50, -40, -30, -20, -10, 0, 10, 20, 30, 40, 50, 60 };

int Eval::evaluate(Board* board)
{
    int score = 0;
    // Material score
    score += materialScore(board);
    // Piece square tables
    score += pieceSquareTables(board);
    // Mobility score
    score += mobilityScore(board);
    // King safety score
    score += kingSafetyScore(board);
    return score;
}

int Eval::materialScore(Board* board)
{
    int score = 0;
    for (int piece = PAWN; piece <= KING; piece++) {
        int pieceCount =popcount(board->bb->pieces[WHITE] & board->bb->piecesByType[piece]);
    score += pieceCount * PieceValues[piece];
    pieceCount = popcount(board->bb->pieces[BLACK] & board->bb->piecesByType[piece]);
    score -= pieceCount * PieceValues[piece];
    }
    return score;
}

int Eval::pieceSquareTables(Board* board)
{
    int score = 0;
    for (int square = 0; square < 64; square++) {
    int piece = board->getPiece(square);
    if (piece != NO_PIECE) {
        int color = (board->bb->pieces[WHITE] & (1ull << square)) ? WHITE : BLACK;
        if (color == WHITE) {
            score += PSQT[piece][square];
        }
        else {
            score -= PSQT[piece][flipSquare(square)];
        }
        }
    }
    return score;
}

int Eval::mobilityScore(Board* board)
{
    int score = 0;
    int mobility = 0;
    mobility += popcount(Movegen::getPseudoLegalMoves(board, WHITE) & ~board->bb->pieces[WHITE]);
    mobility -= popcount(Movegen::getPseudoLegalMoves(board, BLACK) & ~board->bb->pieces[BLACK]);
    for (int piece = KNIGHT; piece <= QUEEN; piece++) {
        mobility += popcount(Movegen::getPseudoLegalMoves(board, WHITE, piece) & ~board->bb->pieces[WHITE]);
        mobility -= popcount(Movegen::getPseudoLegalMoves(board, BLACK, piece) & ~board->bb->pieces[BLACK]);
    }
    switch (popcount(board->bb->piecesByType[KNIGHT] & board->bb->pieces[WHITE])) {
        case 0: break;
        case 1: score += KnightMobility[mobility]; break;
        case 2: score += KnightMobility[mobility] * 2; break;
    }
    switch (popcount(board->bb->piecesByType[BISHOP] & board->bb->pieces[WHITE])) {
        case 0: break;
        case 1: score += BishopMobility[mobility]; break;
        case 2: score += BishopMobility[mobility] * 2; break;
    }
    switch (popcount(board->bb->piecesByType[ROOK] & board->bb->pieces[WHITE])) {
        case 0: break;
        case 1: score += RookMobility[mobility]; break;
        case 2: score += RookMobility[mobility] * 2; break;
    }
    switch (popcount(board->bb->piecesByType[QUEEN] & board->bb->pieces[WHITE])) {
        case 0: break;
        case 1: score += QueenMobility[mobility]; break;
        case 2: score += QueenMobility[mobility] * 2; break;
    }
    mobility = 0;
    mobility -= popcount(Movegen::getPseudoLegalMoves(board, WHITE) & ~board->bb->pieces[WHITE]);
    mobility += popcount(Movegen::getPseudoLegalMoves(board, BLACK) & ~board->bb->pieces[BLACK]);
    for (int piece = KNIGHT; piece <= QUEEN; piece++) {
        mobility -= popcount(Movegen::getPseudoLegalMoves(board, WHITE, piece) & ~board->bb->pieces[WHITE]);
        mobility += popcount(Movegen::getPseudoLegalMoves(board, BLACK, piece) & ~board->bb->pieces[BLACK]);
    }
    switch (popcount(board->bb->piecesByType[KNIGHT] & board->bb->pieces[BLACK])) {
        case 0: break;
        case 1: score -= KnightMobility[mobility]; break;
        case 2: score -= KnightMobility[mobility] * 2; break;
    }
    switch (popcount(board->bb->piecesByType[BISHOP] & board->bb->pieces[BLACK])) {
        case 0: break;
        case 1: score -= BishopMobility[mobility]; break;
        case 2: score -= BishopMobility[mobility] * 2; break;
    }
        switch (popcount(board->bb->piecesByType[ROOK] & board->bb->pieces[BLACK])) {
        case 0: break;
        case 1: score -= RookMobility[mobility]; break;
        case 2: score -= RookMobility[mobility] * 2; break;
    }
    switch (popcount(board->bb->piecesByType[QUEEN] & board->bb->pieces[BLACK])) {
        case 0: break;
        case 1: score -= QueenMobility[mobility]; break;
        case 2: score -= QueenMobility[mobility] * 2; break;
    }

    return score;
}

int Eval::kingSafetyScore(Board* board)
{
    int score = 0;
    int kingSquare = __builtin_ctzll(board->bb->piecesByType[KING] & board->bb->pieces[WHITE]);
    int attacked = popcount(getAttacked(board->bb, ~WHITE) & (board->bb->pieces[WHITE] & ~board->bb->piecesByType[KING]));
    score -= KingSafety[attacked];

    kingSquare = __builtin_ctzll(board->bb->piecesByType[KING] & board->bb->pieces[BLACK]);
    attacked = popcount(getAttacked(board->bb, WHITE) & (board->bb->pieces[BLACK] & ~board->bb->piecesByType[KING]));
    score += KingSafety[attacked];
    return score;
}

unsigned long long getAttacked(Bitboard* bb, int color) {
    unsigned long long occupied = bb->pieces[WHITE] | bb->pieces[BLACK];
    unsigned long long attacks = getRookAttacks(bb->piecesByType[ROOK] | bb->piecesByType[QUEEN], occupied) |
                                 getBishopAttacks(bb->piecesByType[BISHOP] | bb->piecesByType[QUEEN], occupied) |
                                 getPawnAttacks(color, bb->piecesByType[PAWN]);
    return attacks;
}

int popcount(unsigned long long bb) {
    int count;
    for (count = 0; bb; count++) {
        bb &= bb - 1;
    }
    return count;
}

int flipSquare(int square) {
    return 63 - square;
}