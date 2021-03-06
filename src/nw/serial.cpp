/*****************************************************************************/
/*  HEADER INCLUDES                                                          */
/*****************************************************************************/

#include "nw/serial.hpp"

#include <algorithm>
#include <limits>

/*****************************************************************************/
/*  USING DECLERATIONS                                                       */
/*****************************************************************************/

using nw::serial;

/*****************************************************************************/
/*  PUBLIC METHODS                                                           */
/*****************************************************************************/

serial::serial(int match, int miss, int gap)
{
    this->match = match;
    this->miss = miss;
    this->gap = gap;
}

nw::trace& serial::operator()(std::size_t rw, std::size_t cl)
{
    return matrix[rw * n_col + cl];
}

int serial::fill(nw::input const& ref, nw::input const& src)
{
    n_row = src.length();
    n_col = ref.length();

    matrix.resize(n_row * n_col);
    matrix.shrink_to_fit();

    int val = std::numeric_limits<int>::min() - std::min({match, miss, gap});

    std::vector<int> prev(n_col, val);
    std::vector<int> curr(n_col, val);

    for (std::size_t rw = 0; rw < n_row; ++rw)
    {
        std::swap(prev, curr);

        curr[0] = rw * gap;
        (*this)(rw, 0) = nw::trace::insert;

        for (std::size_t cl = 1; cl < n_col; ++cl)
        {
            int pair = prev[cl - 1] + ((ref[cl] == src[rw]) ? match : miss);
            int insert = prev[cl] + gap;
            int remove = curr[cl - 1] + gap;

            curr[cl] = std::max({pair, insert, remove});
            (*this)(rw, cl) = find_trace(pair, insert, remove);
        }
    }

    return curr[n_col - 1];
}

int serial::score(nw::input const& ref, nw::input const& src)
{
    n_row = src.length();
    n_col = ref.length();

    int val = std::numeric_limits<int>::min() - std::min({match, miss, gap});

    std::vector<int> prev(n_col, val);
    std::vector<int> curr(n_col, val);

    for (std::size_t rw = 0; rw < n_row; ++rw)
    {
        std::swap(prev, curr);

        curr[0] = rw * gap;

        for (std::size_t cl = 1; cl < n_col; ++cl)
        {
            int pair = prev[cl - 1] + ((ref[cl] == src[rw]) ? match : miss);
            int insert = prev[cl] + gap;
            int remove = curr[cl - 1] + gap;

            curr[cl] = std::max({pair, insert, remove});
        }
    }

    return curr[n_col - 1];
}

/*****************************************************************************/
/*  PRIVATE METHODS                                                          */
/*****************************************************************************/

nw::trace serial::find_trace(int pair, int insert, int remove)
{
    if (pair > insert)
    {
        return (pair > remove) ? nw::trace::pair : nw::trace::remove;
    }
    else
    {
        return (insert > remove) ? nw::trace::insert : nw::trace::remove;
    }
}
