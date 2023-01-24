// Implementation of move generation

#include "movegen.h"
#include "bitboard.h"
#include "eval.h"

std::vector<Move> Movegen::generateMoves(Board* board)
{
    std::vector<Move> moves;
    unsigned long long pieces = board->bb->pieces[board->toMove];
    while (pieces) {
        int square = __builtin_ctzll(pieces);
        unsigned long long attacks = getAttacks(board->bb, square);
        while (attacks) {
            int to = __builtin_ctzll(attacks);
            moves.push_back(Move(square, to));
            attacks &= attacks - 1;
        }
        pieces &= pieces - 1;
    }
    return moves;
}

std::vector<Move> Movegen::generateMoves(Board* board, int piece)
{
    std::vector<Move> moves;
    unsigned long long pieces = board->bb->piecesByType[piece] & board->bb->pieces[board->toMove];
    while (pieces) {
        int square = __builtin_ctzll(pieces);
        unsigned long long attacks = getAttacks(board->bb, square, piece);
        while (attacks) {
            int to = __builtin_ctzll(attacks);
            moves.push_back(Move(square, to));
            attacks &= attacks - 1;
        }
        pieces &= pieces - 1;
    }
    return moves;
}

unsigned long long Movegen::getPseudoLegalMoves(Board* board, int color)
{
    unsigned long long moves = 0;
    unsigned long long pieces = board->bb->pieces[color];
    while (pieces) {
        int square = __builtin_ctzll(pieces);
        moves |= getAttacks(board->bb, square);
        pieces &= pieces - 1;
    }
    return moves;
}

unsigned long long Movegen::getPseudoLegalMoves(Board* board, int color, int piece)
{
    unsigned long long moves = 0;
    unsigned long long pieces = board->bb->piecesByType[piece] & board->bb->pieces[color];
    while (pieces) {
        int square = __builtin_ctzll(pieces);
        moves |= getAttacks(board->bb, square, piece);
        pieces &= pieces - 1;
    }
    return moves;
}

unsigned long long Movegen::getAttacks(Bitboard* bb, int square, int piece)
{
    unsigned long long attacks = 0;
    switch (piece) {
        case PAWN:
            attacks = getPawnAttacks(bb->pieces[WHITE] & bb->piecesByType[PAWN], square);
            break;
        case KNIGHT:
            attacks = KnightAttacks[square];
            break;
        case BISHOP:
            attacks = getBishopAttacks(bb->piecesByType[BISHOP] | bb->piecesByType[QUEEN], bb->pieces[WHITE] | bb->pieces[BLACK], square);
            break;
        case ROOK:
            attacks = getRookAttacks(bb->piecesByType[ROOK] | bb->piecesByType[QUEEN], bb->pieces[WHITE] | bb->pieces[BLACK], square);
            break;
        case QUEEN:
            attacks = getBishopAttacks(bb->piecesByType[BISHOP] | bb->piecesByType[QUEEN], bb->pieces[WHITE] | bb->pieces[BLACK], square) |
                      getRookAttacks(bb->piecesByType[ROOK] | bb->piecesByType[QUEEN], bb->pieces[WHITE] | bb->pieces[BLACK], square);
            break;
        case KING:
            attacks = KingAttacks[square];
            break;
    }
    return attacks;
}

unsigned long long Movegen::getAttacks(Bitboard* bb, int square)
{
    unsigned long long attacks = 0;
    int piece = getPiece(bb->pieces[WHITE] | bb->pieces[BLACK], square);
    switch (piece) {
        case PAWN:
            attacks = getPawnAttacks(bb->pieces[WHITE] & bb->piecesByType[PAWN], square);
            break;
        case KNIGHT:
            attacks = KnightAttacks[square];
            break;
        case BISHOP:
            attacks = getBishopAttacks(bb->piecesByType[BISHOP] | bb->piecesByType[QUEEN], bb->pieces[WHITE] | bb->pieces[BLACK], square);
            break;
        case ROOK:
            attacks = getRookAttacks(bb->piecesByType[ROOK] | bb->piecesByType[QUEEN], bb->pieces[WHITE] | bb->pieces[BLACK], square);
            break;
        case QUEEN:
            attacks = getBishopAttacks(bb->piecesByType[BISHOP] | bb->piecesBy[QUEEN], bb->pieces[WHITE] | bb->pieces[BLACK], square) |
                      getRookAttacks(bb->piecesByType[ROOK] | bb->piecesByType[QUEEN], bb->pieces[WHITE] | bb->pieces[BLACK], square);
            break;
        case KING:
            attacks = KingAttacks[square];
            break;
    }
    return attacks;
}

unsigned long long Movegen::getPawnAttacks(unsigned long long pawns, int square)
{
    return PawnAttacks[square][!!(pawns & (1ull << square))];
}

int Movegen::getPiece(unsigned long long pieces, int square)
{
    for (int i = 0; i < 6; i++) {
        if (pieces & (1ull << square)) {
            return i;
        }
    }
    return -1;
}

unsigned long long getPawnAttacks(unsigned long long pawns, int square)
{
    unsigned long long attacks = 0;
    if (square < 8) {
        return 0;
    }
    if (pawns & (1ull << (square - 9))) {
        attacks |= 1ull << (square - 9);
    }
    if (pawns & (1ull << (square - 7))) {
        attacks |= 1ull << (square - 7);
    }
    return attacks;
}

unsigned long long getRookAttacks(unsigned long long rooks, unsigned long long occupied, int square)
{
    unsigned long long attacks = 0;
    unsigned long long blockers = occupied & ~((1ull << square) - 1);
    unsigned long long piece = 1ull << square;
    while (blockers) {
        int blockSq = __builtin_ctzll(blockers);
        if (blockSq > square) {
            attacks |= (piece `oaicite:{"index":0,"invalid_reason":"Malformed citation << (blockSq - square)) - 1;\n            break;\n        }\n        attacks |= (piece >>"}` (square - blockSq)) - 1;
        blockers &= blockers - 1;
    }
    blockers = occupied & ((1ull << square) - 1);
    while (blockers) {
        int blockSq = 63 - __builtin_clzll(blockers);
        if (blockSq < square) {
            attacks |= (piece >> (square - blockSq)) - 1;
            break;
        }
        attacks |= (piece << (blockSq - square)) - 1;
        blockers &= blockers - 1;
    }
    return attacks & ~occupied;
    }
}

unsigned long long getBishopAttacks(unsigned long long bishops, unsigned long long occupied, int square)
{
    unsigned long long attacks = 0;
    unsigned long long blockers = occupied & ~((1ull << square) - 1);
    unsigned long long piece = 1ull << square;
    while (blockers) {
        int blockSq = __builtin_ctzll(blockers);
        if (blockSq > square) {
            if (blockSq - square == blockSq % 8 - square % 8) {
                attacks |= (piece `oaicite:{"index":0,"invalid_reason":"Malformed citation << (blockSq - square)) - 1;\n            }\n            break;\n        }\n        if (blockSq - square == blockSq % 8 - square % 8) {\n            attacks |= (piece >>"}` (square - blockSq)) - 1;
        }
        blockers &= blockers - 1;
    }
    blockers = occupied & ((1ull << square) - 1);
    while (blockers) {
        int blockSq = 63 - __builtin_clzll(blockers);
        if (blockSq < square) {
            if (square - blockSq == square % 8 - blockSq % 8) {
                attacks |= (piece >> (square - blockSq)) - 1;
            }
            break;
        }
        if (square - blockSq == square % 8 - blockSq % 8) {
            attacks |= (piece << (blockSq - square)) - 1;
        }
        blockers &= blockers - 1;
    }
    return attacks & ~occupied;
}

int getPiece(unsigned long long pieces, int square)
{
    if (pieces & (1ull `oaicite:{"index":0,"invalid_reason":"Malformed citation << square)) {\n        return 0;\n    }\n    pieces >>= square + 1;\n    int count = 0;\n    while (pieces) {\n        count += pieces & 1;\n        pieces >>"}`= 1;
    }
    return count + 1;
}