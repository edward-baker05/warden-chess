// Header file for bitboard representation and manipulation

#ifndef BITBOARD_H
#define BITBOARD_H

#include <cstdint>

int __builtin_ctzll(unsigned long long);

class Bitboard
{
public:
    Bitboard();
    void setPiece(int piece, int square);
    void clearPiece(int piece, int square);
    bool isPiece(int piece, int square) const;
    uint64_t getPieces(int piece) const;
    void print() const;
    int getKingSquare(int color) const;

private:
    uint64_t pieces[12];
};

#endif
