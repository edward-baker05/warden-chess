// Implementation of the UCI protocol

#include "uci.h"
#include "search.h"
#include <iostream>
#include <string>

void UCI::loop() {
    std::string input;
    while (std::getline(std::cin, input)) {
        if (input == "uci") {
            sendId();
            sendOptions();
        } else if (input == "isready") {
            std::cout << "readyok" << std::endl;
        } else if (input.substr(0, 6) == "position") {
            setPosition(input);
        } else if (input.substr(0, 4) == "go") {
            search();
        } else if (input == "quit") {
            break;
        }
    }
}

void UCI::sendId() {
    std::cout << "id name MyChessEngine" << std::endl;
    std::cout << "id author John Doe" << std::endl;
}

void UCI::sendOptions() {
    std::cout << "option name Hash type spin default 16 min 1 max 65536" << std::endl;
}

void UCI::setPosition(const std::string& input) {
    std::string fen = input.substr(9);
    board->loadFEN(fen);
    tt->clear();
}

void UCI::search() {
    Move bestMove = search(board);
    std::cout << "bestmove " << bestMove.toString() << std::endl;
}