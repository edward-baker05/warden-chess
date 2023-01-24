// Implementation of bitboard representation and manipulation

#include "bitboard.h"

const U64 RookMasks[] = { ... };
const U64 RookMagics[] = { ... };
const int RookShifts[] = { ... };
const U64 RookAttacks[][4096] = { ... };

const U64 BishopMasks[] = { ... };
const U64 BishopMagics[] = { ... };
const int BishopShifts[] = { ... };
const U64 BishopAttacks[][512] = { ... };


Bitboard::Bitboard()
{
    // Initialize all bitboards to 0
    for (int i = 0; i < 12; i++)
        pieces[i] = 0ull;
}

void Bitboard::setPiece(int piece, int square)
{
    pieces[piece] |= (1ull << square);
}

void Bitboard::clearPiece(int piece, int square)
{
    pieces[piece] &= ~(1ull << square);
}

bool Bitboard::isPiece(int piece, int square) const
{
    return (pieces[piece] & (1ull << square)) != 0;
}

uint64_t Bitboard::getPieces(int piece) const
{
    return pieces[piece];
}

void Bitboard::print() const
{
    for (int rank = 7; rank >= 0; rank--)
    {
        for (int file = 0; file < 8; file++)
        {
            int square = rank * 8 + file;
            bool empty = true;
            for (int p = 0; p < 12; p++)
            {
                if (isPiece(p, square))
                {
                    empty = false;
                    std::cout << "P";
                    break;
                }
            }
            if (empty)
                std::cout << "-";
        }
        std::cout << std::endl;
    }
}

int Bitboard::getKingSquare(int color) const
{
    U64 king = pieces[color == WHITE ? WHITE_KING : BLACK_KING];
    return __builtin_ctzll(king);
}

U64 Bitboard::getAttacked(int color) const
{
    U64 attacked = 0;
    // Pawns
    U64 pawns = pieces[color == WHITE ? WHITE_PAWN : BLACK_PAWN];
    attacked |= (color == WHITE ? shift_up_left(pawns) : shift_down_left(pawns));
    attacked |= (color == WHITE ? shift_up_right(pawns) : shift_down_right(pawns));

    // Knights
    U64 knights = pieces[color == WHITE ? WHITE_KNIGHT : BLACK_KNIGHT];
    U64 shifted = shift_up_left(knights);
    shifted |= shift_up_right(knights);
    shifted |= shift_down_left(knights);
    shifted |= shift_down_right(knights);
    shifted |= shift_left(shifted);
    shifted |= shift_right(shifted);
    attacked |= shifted;

    // Bishops and Queens
    U64 bishops = pieces[color == WHITE ? WHITE_BISHOP : BLACK_BISHOP];
    U64 queens = pieces[color == WHITE ? WHITE_QUEEN : BLACK_QUEEN];
    attacked |= getBishopAttacks(bishops | queens, occupied);

    // Rooks and Queens
    U64 rooks = pieces[color == WHITE ? WHITE_ROOK : BLACK_ROOK];
    attacked |= getRookAttacks(rooks | queens, occupied);

    // King
    U64 king = pieces[color == WHITE ? WHITE_KING : BLACK_KING];
    U64 shiftedKing = shift_left(king);
    shiftedKing |= shift_right(king);
    shiftedKing |= shift_up(shiftedKing);
    shiftedKing |= shift_down(shiftedKing);
    attacked |= shiftedKing;

    return attacked;
}

U64 Bitboard::shift_up_left(U64 bb) const
{
    return (bb & ~FILE_A) << 7;
}

U64 Bitboard::shift_up_right(U64 bb) const
{
    return (bb & ~FILE_H) `oaicite:{"index":0,"invalid_reason":"Malformed citation << 9;\n}\n\nU64 Bitboard::shift_down_left(U64 bb) const\n{\n    return (bb & ~FILE_A) >> 9;\n}\n\nU64 Bitboard::shift_down_right(U64 bb) const\n{\n    return (bb & ~FILE_H) >> 7;\n}\n\nU64 Bitboard::shift_left(U64 bb) const\n{\n    return (bb & ~FILE_A) >>"}` 1;
}

U64 Bitboard::shift_right(U64 bb) const
{
    return (bb & ~FILE_H) << 1;
}

U64 Bitboard::shift_up(U64 bb) const
{
    return bb `oaicite:{"index":1,"invalid_reason":"Malformed citation << 8;\n}\n\nU64 Bitboard::shift_down(U64 bb) const\n{\n    return bb >>"}` 8;
}

U64 Bitboard::getBishopAttacks(U64 pieces, U64 occupied) const
{
    U64 result = 0;
    while (pieces)
    {
        int square = __builtin_ctzll(pieces);
        U64 attacks = getBishopAttacks(square, occupied) & ~occupied;
        result |= attacks;
        occupied |= attacks | (1ull << square);
        pieces &= pieces - 1;
    }
    return result;
}

U64 Bitboard::getBishopAttacks(int square, U64 occupied) const
{
    U64 mask = BishopMasks[square];
    U64 occ = occupied & mask;
    U64 magic = BishopMagics[square];
    int index = (int)((occ * magic) >> BishopShifts[square]);
    return BishopAttacks[square][index];
}

U64 Bitboard::getRookAttacks(U64 pieces, U64 occupied) const
{
    U64 result = 0;
    while (pieces)
    {
        int square = __builtin_ctzll(pieces);
        U64 attacks = getRookAttacks(square, occupied) & ~occupied;
        result |= attacks;
        occupied |= attacks | (1ull << square);
        pieces &= pieces - 1;
    }
    return result;
}

U64 Bitboard::getRookAttacks(int square, U64 occupied) const
{
    U64 mask = RookMasks[square];
    U64 occ = occupied & mask;
    U64 magic = RookMagics[square];
    int index = (int)((occ * magic) >> RookShifts[square]);
    return RookAttacks[square][index];
}

void Bitboard::clearBit(int square)
{
    pieces[WHITE] &= ~(1ull << square);
    pieces[BLACK] &= ~(1ull << square);
}

void Bitboard::setBit(int square, int color)
{
    pieces[color] |= (1ull << square);
}

void Bitboard::removePiece(int piece, int color)
{
    pieces[color] &= ~(1ull << piece);
}

void Bitboard::addPiece(int piece, int color)
{
    pieces[color] |= (1ull << piece);
}
