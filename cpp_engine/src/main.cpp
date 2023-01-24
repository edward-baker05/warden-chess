// Entry point of the program

#include <iostream>
#include "board.h"
#include "movegen.h"
#include "search.h"
#include "eval.h"
#include "uci.h"

int main()
{
    Board board;
    Movegen movegen;
    Search search;
    Eval eval;
    Uci uci;

    // Initialize the chess engine
    uci.init(board, movegen, search, eval);

    // Start the UCI loop to communicate with a chess GUI
    uci.loop();

    return 0;
}
