// Header file for transposition table

#ifndef TT_H
#define TT_H

#include <array>
#include <cstdint>

class TT {
public:
    struct Entry {
        std::uint64_t key;
        std::int32_t value;
        std::int32_t depth;
        std::int32_t flags;
    };

    static const std::size_t SIZE = 1 << 20;

    void clear();
    void store(std::uint64_t key, std::int32_t value, std::int32_t depth, std::int32_t flags);
    bool probe(std::uint64_t key, std::int32_t& value, std::int32_t& depth, std::int32_t& flags) const;

private:
    std::array<Entry, SIZE> table;
};

#endif
