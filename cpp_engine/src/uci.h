// Header file for UCI protocol support

#ifndef UCI_H
#define UCI_H

#include <string>
#include "board.h"

class UCI {
public:
    UCI();
    void start();
    void stop();
    void setOption(std::string name, std::string value);
    std::string getOption(std::string name);
    void position(std::string fen);
    void go(std::string movetime);
    void go(std::string depth, std::string movetime);
    void go(std::string depth, std::string moves_to_search, std::string movetime);
    void isready();
    void quit();
    void ucinewgame();
    void perft(std::string depth);

private:
    Board* board;
    bool isRunning;
};

#endif
