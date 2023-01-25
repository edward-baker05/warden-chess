// Implementation of transposition table

#include "tt.h"

TranspositionTable::TranspositionTable(int size) : size(size) {
    // Allocate memory for the transposition table
    table = new TTEntry[size];
}

TranspositionTable::~TranspositionTable() {
    // Free the memory allocated for the transposition table
    delete[] table;
}

void TranspositionTable::store(uint64_t hash, int depth, int value, int type, Move move) {
    // Store the given information in the transposition table
    int index = hash % size;
    TTEntry& entry = table[index];
    entry.hash = hash;
    entry.depth = depth;
    entry.value = value;
    entry.type = type;
    entry.move = move;
}

TTEntry TranspositionTable::lookup(uint64_t hash, int depth) {
    // Look up the information stored for the given hash in the transposition table
    int index = hash % size;
    TTEntry& entry = table[index];
    if (entry.hash == hash && entry.depth >= depth) {
        return entry;
    } else {
        return TTEntry();
    }
}
