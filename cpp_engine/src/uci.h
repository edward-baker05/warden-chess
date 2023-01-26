// Header file for UCI protocol support

#ifndef UCI_H
#define UCI_H

#include <string>

class UCI {
public:
void loop();
void sendId();
void sendOptions();
void setPosition(const std::string& input);
void search();
void init(Board* board, MoveGenerator* movegen, Search* search, Evaluator* eval);
};

#endif // UCI_H