// Implementation of board representation and manipulation

#include "board.h"
#include "bitboard.h"
#include "movegen.h"

Board::Board()
{
    // Initialize the bitboard
    bb = new Bitboard();

    // Initialize the board to the starting position
    reset();
}

Board::~Board()
{
    delete bb;
}

void Board::reset()
{
    // Clear all pieces from the board
    for (int i = 0; i < 12; i++)
        bb->clearPieces(i);

    // Set the pieces to the starting position
    bb->setPiece(WHITE_PAWN, A2);
    bb->setPiece(WHITE_PAWN, B2);
    bb->setPiece(WHITE_PAWN, C2);
    // ...
    bb->setPiece(BLACK_KING, E8);

    // Initialize other variables
    toMove = WHITE;
    castleRights = CASTLE_WHITE_KING | CASTLE_WHITE_QUEEN | CASTLE_BLACK_KING | CASTLE_BLACK_QUEEN;
    enPassantSquare = -1;
    fiftyMoveCounter = 0;
    fullMoveCounter = 1;
    moveList.clear();
    Movegen::generateMoves(this);
}

void Board::makeMove(Move move)
{
    moveList.push_back(move);
    int from = move.from;
    int to = move.to;
    int capturedPiece = move.capturedPiece;
    int promotedPiece = move.promotedPiece;
    U64 fromBB = 1ull << from;
    U64 toBB = 1ull << to;

    // Move the piece to its new position
    bb->clearBit(from);
    bb->setBit(to, toMove);
    if (promotedPiece != NO_PIECE) {
        //If it was a promoted move, remove the pawn and add the promoted piece
        bb->removePiece(PAWN, toMove);
        bb->addPiece(promotedPiece, toMove);
    }
    if (capturedPiece != NO_PIECE) {
        // If a piece was captured, remove it from the board
        bb->removePiece(capturedPiece, ~toMove);
    }

    // Update other variables
    toMove = ~toMove;
    if (capturedPiece == NO_PIECE && getPiece(from) != PAWN) {
        fiftyMoveCounter = 0;
    }
    else {
        fiftyMoveCounter++;
    }
    fullMoveCounter++;
    Movegen::generateMoves(this);
}


void Board::unmakeMove()
{
    if (moveList.empty())
        return;
    Move lastMove = moveList.back();
    int from = lastMove.from;
    int to = lastMove.to;
    int capturedPiece = lastMove.capturedPiece;
    int promotedPiece = lastMove.promotedPiece;
    U64 fromBB = 1ull << from;
    U64 toBB = 1ull << to;

    // Move the piece back to its original position
    bb->setBit(from, toMove);
    bb->clearBit(to);
    if (promotedPiece != NO_PIECE) {
        //If it was a promoted move, remove the promoted piece and add the pawn back
        bb->removePiece(promotedPiece, toMove);
        bb->addPiece(PAWN, toMove);
    }
    if (capturedPiece != NO_PIECE) {
        // If a piece was captured, add it back to the board
        bb->addPiece(capturedPiece, ~toMove);
    }

    // Update other variables
    toMove = ~toMove;
    fiftyMoveCounter++;
    fullMoveCounter--;
    if (capturedPiece == NO_PIECE && getPiece(from) != PAWN) {
        fiftyMoveCounter = 0;
    }

    moveList.pop_back();
}

bool Board::isCheck() const
{
    // get the square of the king
    int king_square = bb->getKingSquare(toMove);
    // get the squares attacked by the opponent
    U64 attacked = bb->getAttacked(~toMove);
    // check if the king is in the attacked squares
    if((attacked & (1ull << king_square)) != 0)
        return true;
    return false;
}


bool Board::isMate() const
{
    if(isCheck()) {
        if(moveList.empty())
            return true;
    }
    return false;
}


bool Board::isStaleMate() const
{
    if(!isCheck()) {
        if(moveList.empty())
            return true;
    }
    return false;
}


void Board::print() const
{
    bb->print();
    std::cout << "Side to move: " << (toMove == WHITE ? "WHITE" : "BLACK") << std::endl;
    std::cout << "Castling rights: ";
    if (castleRights & CASTLE_WHITE_KING) std::cout << "K";
    if (castleRights & CASTLE_WHITE_QUEEN) std::cout << "Q";
    if (castleRights & CASTLE_BLACK_KING) std::cout << "k";
    if (castleRights & CASTLE_BLACK_QUEEN) std::cout << "q";
    std::cout << std::endl;
    std::cout << "En passant square: ";
    if (enPassantSquare == -1) std::cout << "-";
    else std::cout << (char)('a' + (enPassantSquare % 8)) << (enPassantSquare / 8 + 1);
    std::cout << std::endl;
    std::cout << "Fifty move counter: " << fiftyMoveCounter << std::endl;
    std::cout << "Full move counter: " << fullMoveCounter << std::endl;
}


void Board::loadFEN(const std::string& fen)
{
    // Clear all pieces from the board
    for (int i = 0; i < 12; i++)
        bb->clearPieces(i);

    int square = 0;
    int piece = 0;
    int i = 0;
    while (fen[i] != ' ')
    {
        switch (fen[i])
        {
            case 'P': piece = WHITE_PAWN; break;
            case 'N': piece = WHITE_KNIGHT; break;
            case 'B': piece = WHITE_BISHOP; break;
            case 'R': piece = WHITE_ROOK; break;
            case 'Q': piece = WHITE_QUEEN; break;
            case 'K': piece = WHITE_KING; break;
            case 'p': piece = BLACK_PAWN; break;
            case 'n': piece = BLACK_KNIGHT; break;
            case 'b': piece = BLACK_BISHOP; break;
            case 'r': piece = BLACK_ROOK; break;
            case 'q': piece = BLACK_QUEEN; break;
            case 'k': piece = BLACK_KING; break;
            case '/': square -= 16; break;
            case '1': square++; break;
            case '2': square += 2; break;
            case '3': square += 3; break;
            case '4': square += 4; break;
            case '5': square += 5; break;
            case '6': square += 6; break;
            case '7': square += 7; break;
            case '8': square += 8; break;
        }
        if (piece)
        {
            bb->setPiece(piece, square);
            piece = 0;
        }
        i++;
    }
    i++;

    // Load the side to move
    toMove = (fen[i] == 'w') ? WHITE : BLACK;
    i += 2;

    // Load the castling rights
    castleRights = 0;
    while (fen[i] != ' ')
    {
        switch (fen[i])
        {
            case 'K': castleRights |= CASTLE_WHITE_KING; break;
            case 'Q': castleRights |= CASTLE_WHITE_QUEEN; break;
            case 'k': castleRights |= CASTLE_BLACK_KING; break;
            case 'q': castleRights |= CASTLE_BLACK_QUEEN; break;
        }
        i++;
    }
    i++;

    // Load the en passant square
    enPassantSquare = -1;
    if (fen[i] != '-')
    {
        int file = fen[i] - 'a';
        int rank = fen[i + 1] - '1';
        enPassantSquare = rank * 8 + file;
    }

    // Load the fifty move counter and full move counter
    i += 2;
    fiftyMoveCounter = std::stoi(fen.substr(i));

    i = fen.find(" ") + 1;
    fullMoveCounter = std::stoi(fen.substr(i));

}

int Board::getPiece(int square) const
{
    for (int piece = PAWN; piece <= KING; piece++) {
        if (bb->getBit(square, piece)) {
            return piece;
        }
    }
    return NO_PIECE;
}